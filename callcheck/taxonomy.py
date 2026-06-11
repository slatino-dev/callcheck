"""Failure taxonomy and classification for callcheck.

Provides an exhaustive enum of tool-calling failure modes and mapping logic
that turns a list of failed :class:`~callcheck.checkers.CheckResult` objects
into a :class:`FailedCase` with the most-specific label attached.

Failure mode definitions
------------------------
no_call
    Model produced no tool_calls at all (content-only / refusal / stop).
wrong_tool
    A tool call was made but to the wrong function name.
malformed_json
    The ``arguments`` string is not valid JSON (truncation, syntax error).
schema_violation
    Arguments parse as JSON but fail the tool's JSON Schema (wrong property
    types, extra required fields missing, unknown required constraint, etc.).
hallucinated_param
    Arguments contain a key not present in the tool's schema
    (``additionalProperties`` violation — key is not in ``properties``).
missing_required
    A key listed in the schema's ``required`` array is absent from arguments.
type_coercion
    A value has the wrong JSON type (e.g. number instead of string) but is
    otherwise structurally present — often a silent model coercion.
escaping_error
    Arguments JSON has Unicode escape or control-character issues that survive
    ``json.loads`` but produce garbled string values.
truncation
    Arguments JSON is valid JSON but suspiciously short (< 2 chars), or the
    ``arguments`` field is ``None`` / empty string — suggests cut-off output.
parallel_collapse
    Multiple tool calls were expected but the model collapsed them into one
    (or vice versa — one expected, multiple returned).
spurious_call
    More tool calls were returned than expected (unexpected extra calls).
predicate_fail
    A per-argument semantic predicate (enum, range, pattern) was not satisfied.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from callcheck.checkers import CheckResult


class FailureKind(StrEnum):
    """Controlled vocabulary of tool-calling / structured-output failure modes."""

    no_call = "no_call"
    wrong_tool = "wrong_tool"
    malformed_json = "malformed_json"
    schema_violation = "schema_violation"
    hallucinated_param = "hallucinated_param"
    missing_required = "missing_required"
    type_coercion = "type_coercion"
    escaping_error = "escaping_error"
    truncation = "truncation"
    parallel_collapse = "parallel_collapse"
    spurious_call = "spurious_call"
    predicate_fail = "predicate_fail"


# Severity levels: how bad is each failure mode in production?
_SEVERITY: dict[FailureKind, str] = {
    FailureKind.no_call: "critical",
    FailureKind.wrong_tool: "critical",
    FailureKind.malformed_json: "critical",
    FailureKind.schema_violation: "major",
    FailureKind.hallucinated_param: "major",
    FailureKind.missing_required: "major",
    FailureKind.type_coercion: "minor",
    FailureKind.escaping_error: "minor",
    FailureKind.truncation: "major",
    FailureKind.parallel_collapse: "major",
    FailureKind.spurious_call: "minor",
    FailureKind.predicate_fail: "minor",
}

# Priority order used by classify_results when multiple failures are present.
# Lower index = higher priority (the first matched label wins).
_PRIORITY: list[FailureKind] = [
    FailureKind.no_call,
    FailureKind.malformed_json,
    FailureKind.truncation,
    FailureKind.wrong_tool,
    FailureKind.missing_required,
    FailureKind.schema_violation,
    FailureKind.hallucinated_param,
    FailureKind.type_coercion,
    FailureKind.escaping_error,
    FailureKind.parallel_collapse,
    FailureKind.spurious_call,
    FailureKind.predicate_fail,
]

# Human-readable descriptions for every tag (used in reports).
TAGS: dict[str, str] = {k.value: f"Failure mode: {k.value.replace('_', ' ')}" for k in FailureKind}


def severity(kind: FailureKind) -> str:
    """Return the severity string (``"critical"``, ``"major"``, or ``"minor"``)."""
    return _SEVERITY[kind]


class FailedCase:
    """A fully classified failure with the raw model output attached.

    Attributes
    ----------
    task_id:
        The :attr:`~callcheck.tasks.TaskSpec.id` of the task that failed.
    kind:
        The most-specific :class:`FailureKind` label for this failure.
    detail:
        A human-readable explanation of what went wrong.
    raw_output:
        The raw model response dict (the whole ``chat.completion`` object).
    all_kinds:
        Every :class:`FailureKind` that was detected (not just the primary).
    """

    def __init__(
        self,
        task_id: str,
        kind: FailureKind,
        detail: str,
        raw_output: dict[str, Any],
        all_kinds: list[FailureKind] | None = None,
    ) -> None:
        self.task_id = task_id
        self.kind = kind
        self.detail = detail
        self.raw_output = raw_output
        self.all_kinds: list[FailureKind] = all_kinds or [kind]

    def __repr__(self) -> str:
        return f"FailedCase(task_id={self.task_id!r}, kind={self.kind!r})"


def classify_results(check_results: list[CheckResult]) -> list[FailureKind]:
    """Map a list of :class:`~callcheck.checkers.CheckResult` to failure kinds.

    Returns the list of unique failure kinds found, ordered by priority
    (most severe first).  An empty list means all checks passed.
    """
    found: set[FailureKind] = set()
    for result in check_results:
        if not result.passed and result.failure_kind is not None:
            found.add(result.failure_kind)
    # Return in priority order
    return [k for k in _PRIORITY if k in found]


def primary_kind(kinds: list[FailureKind]) -> FailureKind | None:
    """Return the highest-priority failure kind, or None if the list is empty."""
    return kinds[0] if kinds else None
