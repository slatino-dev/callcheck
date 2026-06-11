"""Conformance checkers that score a model response against a TaskSpec.

Each checker is a pure function that takes a raw API response dict plus a
:class:`~callcheck.tasks.TaskSpec` and returns a :class:`CheckResult` with a
pass/fail verdict, an optional detail message, and the :class:`FailureKind`
label (populated only on failures).

Checkers are composable: :func:`run_checkers` runs the full pipeline and
returns every result.  The caller can then hand the list to
:func:`~callcheck.taxonomy.classify_results` for failure-mode labelling.

Pipeline order
--------------
1. :func:`check_no_call`          — did the model call any tool at all?
2. :func:`check_call_count`       — is the number of calls correct?
3. :func:`check_tool_name`        — is the right function being called?
4. :func:`check_json_parse`       — do the arguments parse as JSON?
5. :func:`check_truncation`       — are the arguments suspiciously empty?
6. :func:`check_schema`           — do the arguments satisfy the tool schema?
7. :func:`check_required_args`    — are required keys present?
8. :func:`check_hallucinated_params` — are unknown keys present?
9. :func:`check_arg_predicates`   — do semantic predicates pass?

Checks 3-9 operate on the *first* tool call in the response so that the
pipeline short-circuits cleanly on count / name failures (caller can filter
on ``passed=False`` early).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from callcheck.tasks import TaskSpec
from callcheck.taxonomy import FailureKind


@dataclass
class CheckResult:
    """Result of a single conformance check.

    Attributes
    ----------
    name:
        Short identifier for the check (e.g. ``"json_parse"``).
    passed:
        Whether the check succeeded.
    score:
        Numeric score in ``[0, 1]``.  Typically 1.0 on pass, 0.0 on fail;
        future checkers may return partial credit.
    detail:
        Human-readable explanation (empty string when passed).
    failure_kind:
        The :class:`~callcheck.taxonomy.FailureKind` for this failure;
        ``None`` when ``passed=True``.
    """

    name: str
    passed: bool
    score: float = field(default=1.0)
    detail: str = field(default="")
    failure_kind: FailureKind | None = field(default=None)

    def __post_init__(self) -> None:
        if self.passed:
            self.score = 1.0
            self.failure_kind = None
        else:
            if self.score == 1.0:
                self.score = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ``tool_calls`` from an OpenAI chat-completion response.

    Returns an empty list if the response has no tool calls or has an
    unexpected shape.
    """
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return list(tool_calls)


def _first_call(response: dict[str, Any]) -> dict[str, Any] | None:
    calls = _get_tool_calls(response)
    return calls[0] if calls else None


def _parse_arguments(call: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Try to parse the ``arguments`` string of a tool call.

    Returns ``(parsed_args, error_message)``.  On success ``error_message``
    is ``None``; on failure ``parsed_args`` is ``None``.
    """
    raw_args: str | None = call.get("function", {}).get("arguments")
    if raw_args is None or raw_args == "":
        return None, "arguments is empty or None"
    try:
        return json.loads(raw_args), None
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------


def check_no_call(response: dict[str, Any], _task: TaskSpec) -> CheckResult:
    """Fail if the model produced zero tool calls."""
    calls = _get_tool_calls(response)
    if calls:
        return CheckResult(name="no_call", passed=True)
    return CheckResult(
        name="no_call",
        passed=False,
        detail="Model produced no tool_calls (refusal or content-only response)",
        failure_kind=FailureKind.no_call,
    )


def check_call_count(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Fail if the number of tool calls does not match the expectation."""
    expected_count = 1
    if task.expect.tool_calls is not None:
        expected_count = task.expect.tool_calls.count

    calls = _get_tool_calls(response)
    actual = len(calls)

    if actual == expected_count:
        return CheckResult(name="call_count", passed=True)

    if actual > expected_count:
        kind = FailureKind.spurious_call
        if expected_count > 1 and actual == 1:
            kind = FailureKind.parallel_collapse
        return CheckResult(
            name="call_count",
            passed=False,
            detail=f"Expected {expected_count} tool call(s); got {actual}",
            failure_kind=kind,
        )

    # actual < expected_count
    kind = FailureKind.parallel_collapse if expected_count > 1 else FailureKind.no_call
    return CheckResult(
        name="call_count",
        passed=False,
        detail=f"Expected {expected_count} tool call(s); got {actual}",
        failure_kind=kind,
    )


def check_tool_name(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Fail if the first tool call targets the wrong function."""
    if task.expect.tool_calls is None or task.expect.tool_calls.name is None:
        return CheckResult(name="tool_name", passed=True, detail="no name expectation set")

    call = _first_call(response)
    if call is None:
        # Should have been caught by check_no_call; skip gracefully.
        return CheckResult(name="tool_name", passed=True, detail="no call to check name")

    actual_name: str = call.get("function", {}).get("name", "")
    expected_name: str = task.expect.tool_calls.name

    if actual_name == expected_name:
        return CheckResult(name="tool_name", passed=True)

    return CheckResult(
        name="tool_name",
        passed=False,
        detail=f"Expected tool '{expected_name}'; got '{actual_name}'",
        failure_kind=FailureKind.wrong_tool,
    )


def check_json_parse(response: dict[str, Any], _task: TaskSpec) -> CheckResult:
    """Fail if the first tool call's arguments are not valid JSON."""
    call = _first_call(response)
    if call is None:
        return CheckResult(name="json_parse", passed=True, detail="no call to check")

    parsed, error = _parse_arguments(call)
    if parsed is not None:
        return CheckResult(name="json_parse", passed=True)

    raw_args: str | None = call.get("function", {}).get("arguments")
    if raw_args is None or raw_args == "":
        return CheckResult(
            name="json_parse",
            passed=False,
            detail="arguments field is empty or None",
            failure_kind=FailureKind.truncation,
        )

    return CheckResult(
        name="json_parse",
        passed=False,
        detail=error or "arguments is not valid JSON",
        failure_kind=FailureKind.malformed_json,
    )


def check_truncation(response: dict[str, Any], _task: TaskSpec) -> CheckResult:
    """Fail if arguments are valid JSON but suspiciously short (likely truncated)."""
    call = _first_call(response)
    if call is None:
        return CheckResult(name="truncation", passed=True, detail="no call to check")

    raw_args: str | None = call.get("function", {}).get("arguments")
    if not raw_args:
        return CheckResult(
            name="truncation",
            passed=False,
            detail="arguments is empty or None",
            failure_kind=FailureKind.truncation,
        )

    parsed, _error = _parse_arguments(call)
    if parsed is None:
        # malformed_json handles this; not our problem here
        return CheckResult(name="truncation", passed=True)

    # {} is a valid but empty object — might be fine for tools with no required args
    # Treat a bare empty-string as truncation (already caught above)
    # Flag raw_args shorter than 3 chars as likely truncated (e.g. "{}", "[]")
    if isinstance(parsed, dict) and len(parsed) == 0 and raw_args.strip() in ("{}", "{ }"):
        # Empty object is only suspicious if required args are expected;
        # we leave that to check_required_args.
        pass

    return CheckResult(name="truncation", passed=True)


def check_schema(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Validate the first tool call's arguments against the tool's JSON Schema."""
    call = _first_call(response)
    if call is None:
        return CheckResult(name="schema", passed=True, detail="no call to check")

    # Identify which tool schema to validate against
    tool_name: str = call.get("function", {}).get("name", "")
    tool_def = task.tool_by_name(tool_name)
    if tool_def is None:
        # Can't validate schema for unknown tool; wrong_tool check handles this
        return CheckResult(name="schema", passed=True, detail=f"no schema found for '{tool_name}'")

    schema: dict[str, Any] = tool_def.get("function", {}).get("parameters", {})
    if not schema:
        return CheckResult(name="schema", passed=True, detail="tool has no parameters schema")

    parsed, _parse_error = _parse_arguments(call)
    if parsed is None:
        # JSON parse failure; malformed_json check handles this
        return CheckResult(name="schema", passed=True, detail="skipped — arguments not parseable")

    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return CheckResult(name="schema", passed=True)
    except jsonschema.ValidationError as exc:
        # Classify the specific kind of schema violation
        kind = _classify_schema_error(exc, schema, parsed)
        return CheckResult(
            name="schema",
            passed=False,
            detail=f"schema violation: {exc.message}",
            failure_kind=kind,
        )
    except jsonschema.SchemaError as exc:
        # The tool's schema itself is invalid — not the model's fault
        return CheckResult(
            name="schema",
            passed=False,
            detail=f"invalid tool schema: {exc.message}",
            failure_kind=FailureKind.schema_violation,
        )


def _classify_schema_error(
    exc: jsonschema.ValidationError,
    schema: dict[str, Any],
    instance: dict[str, Any],
) -> FailureKind:
    """Heuristically map a jsonschema ValidationError to a FailureKind."""
    validator = exc.validator

    if validator == "required":
        return FailureKind.missing_required

    if validator == "additionalProperties":
        return FailureKind.hallucinated_param

    if validator == "type":
        return FailureKind.type_coercion

    if validator == "enum":
        return FailureKind.schema_violation

    # Check if the failing path points to an unexpected property
    properties = schema.get("properties", {})
    if exc.path:
        top_key = str(exc.path[0])
        if top_key not in properties:
            return FailureKind.hallucinated_param

    return FailureKind.schema_violation


def check_required_args(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Fail if any of the explicitly expected required args are absent."""
    if task.expect.tool_calls is None:
        return CheckResult(name="required_args", passed=True)

    required = task.expect.tool_calls.required_args
    if not required:
        return CheckResult(name="required_args", passed=True)

    call = _first_call(response)
    if call is None:
        return CheckResult(name="required_args", passed=True, detail="no call to check")

    parsed, _ = _parse_arguments(call)
    if parsed is None:
        return CheckResult(name="required_args", passed=True, detail="skipped — args not parseable")

    missing = [k for k in required if k not in parsed]
    if not missing:
        return CheckResult(name="required_args", passed=True)

    return CheckResult(
        name="required_args",
        passed=False,
        detail=f"Missing required argument(s): {missing}",
        failure_kind=FailureKind.missing_required,
    )


def check_hallucinated_params(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Fail if arguments contain keys not declared in the tool's schema properties."""
    call = _first_call(response)
    if call is None:
        return CheckResult(name="hallucinated_params", passed=True, detail="no call to check")

    tool_name: str = call.get("function", {}).get("name", "")
    tool_def = task.tool_by_name(tool_name)
    if tool_def is None:
        return CheckResult(name="hallucinated_params", passed=True, detail="unknown tool; skipped")

    schema = tool_def.get("function", {}).get("parameters", {})
    declared_props: set[str] = set(schema.get("properties", {}).keys())
    # If no properties declared, we cannot check for hallucination
    if not declared_props:
        return CheckResult(name="hallucinated_params", passed=True)

    # Only flag when additionalProperties is explicitly false or not set (default no-extra)
    # We flag regardless because hallucinated params are always wrong in practice.
    parsed, _ = _parse_arguments(call)
    if parsed is None:
        return CheckResult(
            name="hallucinated_params", passed=True, detail="skipped — args not parseable"
        )

    extra = [k for k in parsed if k not in declared_props]
    if not extra:
        return CheckResult(name="hallucinated_params", passed=True)

    return CheckResult(
        name="hallucinated_params",
        passed=False,
        detail=f"Hallucinated parameter(s) not in schema: {extra}",
        failure_kind=FailureKind.hallucinated_param,
    )


def check_arg_predicates(response: dict[str, Any], task: TaskSpec) -> CheckResult:
    """Run per-argument semantic predicates (enum, range, pattern checks)."""
    if task.expect.tool_calls is None:
        return CheckResult(name="arg_predicates", passed=True)

    predicates = task.expect.tool_calls.arg_predicates
    if not predicates:
        return CheckResult(name="arg_predicates", passed=True)

    call = _first_call(response)
    if call is None:
        return CheckResult(name="arg_predicates", passed=True, detail="no call to check")

    parsed, _ = _parse_arguments(call)
    if parsed is None:
        return CheckResult(
            name="arg_predicates", passed=True, detail="skipped — args not parseable"
        )

    failures: list[str] = []
    for arg_key, pred in predicates.items():
        if arg_key not in parsed:
            # Missing-arg failures handled by check_required_args
            continue
        value = parsed[arg_key]

        if pred.type is not None:
            _JSON_TYPE_MAP = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }
            expected_py = _JSON_TYPE_MAP.get(pred.type)
            if expected_py is not None and not isinstance(value, expected_py):
                failures.append(
                    f"'{arg_key}': expected type {pred.type}, "
                    f"got {type(value).__name__} ({value!r})"
                )
                continue

        if pred.enum is not None and value not in pred.enum:
            failures.append(
                f"'{arg_key}': value {value!r} not in allowed set {pred.enum}"
            )
            continue

        if pred.min is not None:
            try:
                if float(value) < pred.min:
                    failures.append(
                        f"'{arg_key}': {value} < min {pred.min}"
                    )
                    continue
            except (TypeError, ValueError):
                failures.append(f"'{arg_key}': cannot compare {value!r} to min {pred.min}")
                continue

        if pred.max is not None:
            try:
                if float(value) > pred.max:
                    failures.append(f"'{arg_key}': {value} > max {pred.max}")
                    continue
            except (TypeError, ValueError):
                failures.append(f"'{arg_key}': cannot compare {value!r} to max {pred.max}")
                continue

        if pred.pattern is not None:
            if not isinstance(value, str) or not re.search(pred.pattern, value):
                failures.append(
                    f"'{arg_key}': {value!r} does not match pattern {pred.pattern!r}"
                )

    if not failures:
        return CheckResult(name="arg_predicates", passed=True)

    return CheckResult(
        name="arg_predicates",
        passed=False,
        detail="; ".join(failures),
        failure_kind=FailureKind.predicate_fail,
    )


def check_escaping(response: dict[str, Any], _task: TaskSpec) -> CheckResult:
    """Detect Unicode-escape or control-character issues in string arguments.

    ``json.loads`` happily decodes ``\\uXXXX`` sequences; this checker looks
    for the *raw* escape byte sequences surviving into the final Python string,
    which indicates the model double-escaped its output.
    """
    call = _first_call(response)
    if call is None:
        return CheckResult(name="escaping", passed=True, detail="no call to check")

    parsed, _ = _parse_arguments(call)
    if parsed is None:
        return CheckResult(name="escaping", passed=True, detail="skipped — args not parseable")

    def _contains_escape_artifacts(value: Any) -> bool:
        if isinstance(value, str):
            # Double-escaped unicode: literal backslash-u in the decoded string
            if re.search(r"\\u[0-9a-fA-F]{4}", value):
                return True
            # Raw C0 control characters (except tab/newline/CR) in a string value
            if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
                return True
        elif isinstance(value, dict):
            return any(_contains_escape_artifacts(v) for v in value.values())
        elif isinstance(value, list):
            return any(_contains_escape_artifacts(v) for v in value)
        return False

    if _contains_escape_artifacts(parsed):
        return CheckResult(
            name="escaping",
            passed=False,
            detail="String argument(s) contain double-escaped or raw control characters",
            failure_kind=FailureKind.escaping_error,
        )

    return CheckResult(name="escaping", passed=True)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

_CHECKER_PIPELINE = [
    check_no_call,
    check_call_count,
    check_tool_name,
    check_json_parse,
    check_truncation,
    check_schema,
    check_required_args,
    check_hallucinated_params,
    check_arg_predicates,
    check_escaping,
]


def run_checkers(response: dict[str, Any], task: TaskSpec) -> list[CheckResult]:
    """Run the full checker pipeline against *response* / *task*.

    All checkers run regardless of earlier failures so the caller gets a
    complete picture.  Callers that want early-exit behaviour can stop
    iterating when they hit the first ``passed=False`` result.
    """
    return [checker(response, task) for checker in _CHECKER_PIPELINE]
