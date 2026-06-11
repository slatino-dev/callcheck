"""Unit tests for checkers.py and taxonomy.py.

Every test feeds a known-bad (or known-good) payload directly into the
checker functions and asserts the exact FailureKind label.  No network, no
subprocess — this proves the scoring rig itself is correct.
"""

from __future__ import annotations

import json

import pytest

from callcheck.checkers import (
    CheckResult,
    check_arg_predicates,
    check_call_count,
    check_escaping,
    check_hallucinated_params,
    check_json_parse,
    check_no_call,
    check_required_args,
    check_schema,
    check_tool_name,
    run_checkers,
)
from callcheck.tasks import ArgPredicate, Expectation, TaskSpec, ToolCallExpectation
from callcheck.taxonomy import FailureKind, classify_results, primary_kind, severity

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_tool(
    name: str = "get_weather",
    required: list[str] | None = None,
    properties: dict | None = None,
) -> dict:
    if properties is None:
        properties = {
            "location": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        }
    if required is None:
        required = ["location"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _make_task(
    tool_name: str = "get_weather",
    required_args: list[str] | None = None,
    count: int = 1,
    tools: list[dict] | None = None,
    arg_predicates: dict | None = None,
) -> TaskSpec:
    if tools is None:
        tools = [_make_tool(tool_name)]
    if required_args is None:
        required_args = ["location"]

    tc_expect = ToolCallExpectation(
        count=count,
        name=tool_name,
        required_args=required_args,
        arg_predicates={
            k: ArgPredicate(**v) for k, v in (arg_predicates or {}).items()
        },
    )
    return TaskSpec(
        id="test_task",
        messages=[{"role": "user", "content": "test"}],
        tools=tools,
        expect=Expectation(tool_calls=tc_expect),
        tags=["test"],
    )


def _make_call(name: str = "get_weather", arguments: str | None = None) -> dict:
    if arguments is None:
        arguments = json.dumps({"location": "Dublin", "units": "celsius"})
    return {
        "id": "call_abc123",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_completion(
    tool_calls: list[dict] | None = None,
    content: str | None = None,
) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


# ---------------------------------------------------------------------------
# check_no_call
# ---------------------------------------------------------------------------


class TestCheckNoCall:
    def test_passes_when_tool_call_present(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call()])
        result = check_no_call(resp, task)
        assert result.passed is True
        assert result.failure_kind is None

    def test_fails_on_content_only_response(self) -> None:
        task = _make_task()
        resp = _make_completion(content="I cannot help with that.")
        result = check_no_call(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.no_call

    def test_fails_on_empty_tool_calls_list(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[])
        result = check_no_call(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.no_call

    def test_fails_on_null_tool_calls(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=None)
        result = check_no_call(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.no_call

    def test_score_is_zero_on_fail(self) -> None:
        task = _make_task()
        resp = _make_completion(content="no")
        result = check_no_call(resp, task)
        assert result.score == 0.0

    def test_passes_content_only_when_count_is_zero(self) -> None:
        """A zero-call response must PASS when the task expects count=0."""
        task = _make_task(count=0, required_args=[])
        resp = _make_completion(content="Latency is the delay in data transmission.")
        result = check_no_call(resp, task)
        assert result.passed is True
        assert result.failure_kind is None

    def test_fails_when_call_present_but_count_is_zero(self) -> None:
        """A tool call on a count=0 task is NOT caught here; check_call_count handles it."""
        task = _make_task(count=0, required_args=[])
        resp = _make_completion(tool_calls=[_make_call()])
        # check_no_call sees calls present → passes (spurious-call detection
        # is the job of check_call_count, not check_no_call)
        result = check_no_call(resp, task)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_call_count
# ---------------------------------------------------------------------------


class TestCheckCallCount:
    def test_passes_for_correct_single_count(self) -> None:
        task = _make_task(count=1)
        resp = _make_completion(tool_calls=[_make_call()])
        assert check_call_count(resp, task).passed is True

    def test_fails_spurious_extra_calls(self) -> None:
        task = _make_task(count=1)
        resp = _make_completion(tool_calls=[_make_call(), _make_call()])
        result = check_call_count(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.spurious_call

    def test_fails_parallel_collapse_too_few(self) -> None:
        task = _make_task(count=2)
        resp = _make_completion(tool_calls=[_make_call()])
        result = check_call_count(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.parallel_collapse

    def test_passes_for_multi_call_match(self) -> None:
        task = _make_task(count=2)
        resp = _make_completion(tool_calls=[_make_call(), _make_call()])
        assert check_call_count(resp, task).passed is True


# ---------------------------------------------------------------------------
# check_tool_name
# ---------------------------------------------------------------------------


class TestCheckToolName:
    def test_passes_correct_name(self) -> None:
        task = _make_task(tool_name="get_weather")
        resp = _make_completion(tool_calls=[_make_call("get_weather")])
        assert check_tool_name(resp, task).passed is True

    def test_fails_wrong_name(self) -> None:
        task = _make_task(tool_name="get_weather")
        resp = _make_completion(tool_calls=[_make_call("send_email")])
        result = check_tool_name(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.wrong_tool

    def test_passes_when_no_name_expectation(self) -> None:
        task = TaskSpec(
            id="t",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            expect=Expectation(tool_calls=ToolCallExpectation(count=1)),
        )
        resp = _make_completion(tool_calls=[_make_call("anything")])
        assert check_tool_name(resp, task).passed is True

    def test_wrong_name_detail_contains_both_names(self) -> None:
        task = _make_task(tool_name="get_weather")
        resp = _make_completion(tool_calls=[_make_call("search_web")])
        result = check_tool_name(resp, task)
        assert "get_weather" in result.detail
        assert "search_web" in result.detail


# ---------------------------------------------------------------------------
# check_json_parse
# ---------------------------------------------------------------------------


class TestCheckJsonParse:
    def test_passes_valid_json(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments='{"location": "Dublin"}')])
        assert check_json_parse(resp, task).passed is True

    def test_fails_truncated_json(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments='{"location": "Dublin"')])
        result = check_json_parse(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.malformed_json

    def test_fails_empty_string_args(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments="")])
        result = check_json_parse(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.truncation

    def test_fails_none_args(self) -> None:
        task = _make_task()
        call = _make_call()
        call["function"]["arguments"] = None
        resp = _make_completion(tool_calls=[call])
        result = check_json_parse(resp, task)
        assert result.passed is False

    def test_fails_syntax_error(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments='{location: Dublin}')])
        result = check_json_parse(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.malformed_json

    def test_fails_bare_string_not_object(self) -> None:
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments='"Dublin"')])
        # Valid JSON but the downstream schema check will catch type; json_parse passes
        assert check_json_parse(resp, task).passed is True

    def test_fails_unicode_escape_truncation(self) -> None:
        """Truncated mid-unicode-escape should be malformed_json."""
        task = _make_task()
        resp = _make_completion(tool_calls=[_make_call(arguments='{"location": "\\u004')])
        result = check_json_parse(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.malformed_json


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_passes_valid_args(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "units": "celsius"}')]
        )
        assert check_schema(resp, task).passed is True

    def test_fails_wrong_type_for_location(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": 42}')]
        )
        result = check_schema(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.type_coercion

    def test_fails_invalid_enum_value(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "units": "kelvin"}')]
        )
        result = check_schema(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.schema_violation

    def test_fails_missing_required_via_schema(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"units": "celsius"}')]
        )
        result = check_schema(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.missing_required

    def test_passes_when_no_schema(self) -> None:
        """A tool with no parameters schema should not fail schema check."""
        tool_no_schema = {
            "type": "function",
            "function": {"name": "ping", "description": "ping"},
        }
        task = TaskSpec(
            id="t",
            messages=[{"role": "user", "content": "ping"}],
            tools=[tool_no_schema],
            expect=Expectation(tool_calls=ToolCallExpectation(name="ping")),
        )
        resp = _make_completion(tool_calls=[_make_call("ping", arguments="{}")])
        assert check_schema(resp, task).passed is True

    def test_location_as_number_is_type_coercion(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": 3.14}')]
        )
        result = check_schema(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.type_coercion


# ---------------------------------------------------------------------------
# check_required_args
# ---------------------------------------------------------------------------


class TestCheckRequiredArgs:
    def test_passes_when_required_present(self) -> None:
        task = _make_task(required_args=["location"])
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        assert check_required_args(resp, task).passed is True

    def test_fails_when_required_missing(self) -> None:
        task = _make_task(required_args=["location"])
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"units": "celsius"}')]
        )
        result = check_required_args(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.missing_required
        assert "location" in result.detail

    def test_fails_multiple_missing(self) -> None:
        task = _make_task(required_args=["location", "units"])
        resp = _make_completion(tool_calls=[_make_call(arguments="{}")])
        result = check_required_args(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.missing_required

    def test_passes_when_no_required_args_expected(self) -> None:
        task = _make_task(required_args=[])
        resp = _make_completion(tool_calls=[_make_call(arguments="{}")])
        assert check_required_args(resp, task).passed is True


# ---------------------------------------------------------------------------
# check_hallucinated_params
# ---------------------------------------------------------------------------


class TestCheckHallucinatedParams:
    def test_passes_when_no_extra_keys(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        assert check_hallucinated_params(resp, task).passed is True

    def test_fails_on_extra_key(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "timezone": "UTC"}')]
        )
        result = check_hallucinated_params(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.hallucinated_param
        assert "timezone" in result.detail

    def test_fails_multiple_hallucinated_keys(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[
                _make_call(
                    arguments='{"location": "Dublin", "foo": 1, "bar": 2}'
                )
            ]
        )
        result = check_hallucinated_params(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.hallucinated_param

    def test_passes_when_tool_has_no_properties(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object"},
            },
        }
        task = TaskSpec(
            id="t",
            messages=[{"role": "user", "content": "ping"}],
            tools=[tool],
            expect=Expectation(tool_calls=ToolCallExpectation(name="ping")),
        )
        resp = _make_completion(tool_calls=[_make_call("ping", arguments='{"anything": 1}')])
        assert check_hallucinated_params(resp, task).passed is True


# ---------------------------------------------------------------------------
# check_arg_predicates
# ---------------------------------------------------------------------------


class TestCheckArgPredicates:
    def test_passes_valid_enum(self) -> None:
        task = _make_task(
            arg_predicates={"units": {"enum": ["celsius", "fahrenheit"]}}
        )
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "units": "celsius"}')]
        )
        assert check_arg_predicates(resp, task).passed is True

    def test_fails_wrong_enum(self) -> None:
        task = _make_task(
            arg_predicates={"units": {"enum": ["celsius", "fahrenheit"]}}
        )
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "units": "kelvin"}')]
        )
        result = check_arg_predicates(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.predicate_fail
        assert "kelvin" in result.detail

    def test_fails_wrong_type(self) -> None:
        task = _make_task(arg_predicates={"location": {"type": "string"}})
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": 123}')]
        )
        result = check_arg_predicates(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.predicate_fail

    def test_fails_range_min(self) -> None:
        tool = _make_tool(
            name="set_temp",
            required=["value"],
            properties={"value": {"type": "number"}},
        )
        task = TaskSpec(
            id="t",
            messages=[{"role": "user", "content": "set temp"}],
            tools=[tool],
            expect=Expectation(
                tool_calls=ToolCallExpectation(
                    name="set_temp",
                    arg_predicates={"value": ArgPredicate(min=0.0)},
                )
            ),
        )
        resp = _make_completion(
            tool_calls=[_make_call("set_temp", arguments='{"value": -5}')]
        )
        result = check_arg_predicates(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.predicate_fail

    def test_fails_range_max(self) -> None:
        tool = _make_tool(
            name="set_temp",
            required=["value"],
            properties={"value": {"type": "number"}},
        )
        task = TaskSpec(
            id="t",
            messages=[{"role": "user", "content": "set temp"}],
            tools=[tool],
            expect=Expectation(
                tool_calls=ToolCallExpectation(
                    name="set_temp",
                    arg_predicates={"value": ArgPredicate(max=100.0)},
                )
            ),
        )
        resp = _make_completion(
            tool_calls=[_make_call("set_temp", arguments='{"value": 200}')]
        )
        result = check_arg_predicates(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.predicate_fail

    def test_passes_pattern_match(self) -> None:
        task = _make_task(
            arg_predicates={"location": {"pattern": r"^[A-Z][a-z]+"}}
        )
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        assert check_arg_predicates(resp, task).passed is True

    def test_fails_pattern_no_match(self) -> None:
        task = _make_task(
            arg_predicates={"location": {"pattern": r"^\d{5}$"}}
        )
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        result = check_arg_predicates(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.predicate_fail

    def test_skips_missing_key_in_predicates(self) -> None:
        """Predicates for absent keys are skipped (missing_required handles them)."""
        task = _make_task(
            arg_predicates={"units": {"enum": ["celsius", "fahrenheit"]}}
        )
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        # No 'units' key — should pass here; required check would catch absence
        assert check_arg_predicates(resp, task).passed is True


# ---------------------------------------------------------------------------
# check_escaping
# ---------------------------------------------------------------------------


class TestCheckEscaping:
    def test_passes_clean_string(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"}')]
        )
        assert check_escaping(resp, task).passed is True

    def test_fails_double_escaped_unicode(self) -> None:
        """Literal \\u0044 in decoded string = double escaping."""
        task = _make_task()
        # Build a response where the JSON *value* contains a literal backslash-u
        args = json.dumps({"location": "\\u0044ublin"})  # backslash-u in value
        resp = _make_completion(tool_calls=[_make_call(arguments=args)])
        result = check_escaping(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.escaping_error

    def test_fails_control_character_in_string(self) -> None:
        """A raw BEL (0x07) character inside a JSON string value."""
        task = _make_task()
        # json.dumps will escape \x07 as \\u0007, but we craft a payload
        # where the decoded string contains the raw byte.
        bad_value = "Dublin\x07city"
        args = json.dumps({"location": bad_value})
        resp = _make_completion(tool_calls=[_make_call(arguments=args)])
        result = check_escaping(resp, task)
        assert result.passed is False
        assert result.failure_kind == FailureKind.escaping_error

    def test_passes_legitimate_newline_in_string(self) -> None:
        """\\n (0x0a) is explicitly allowed (not in the blocked set)."""
        task = _make_task()
        args = json.dumps({"location": "Dublin\nCity"})
        resp = _make_completion(tool_calls=[_make_call(arguments=args)])
        assert check_escaping(resp, task).passed is True


# ---------------------------------------------------------------------------
# run_checkers — integration
# ---------------------------------------------------------------------------


class TestRunCheckers:
    def test_all_pass_for_good_response(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin", "units": "celsius"}')]
        )
        results = run_checkers(resp, task)
        failed = [r for r in results if not r.passed]
        assert failed == [], f"Unexpected failures: {[r.name for r in failed]}"

    def test_refusal_triggers_no_call(self) -> None:
        task = _make_task()
        resp = _make_completion(content="I cannot help with that.")
        results = run_checkers(resp, task)
        kinds = [r.failure_kind for r in results if not r.passed]
        assert FailureKind.no_call in kinds

    def test_malformed_json_triggers_malformed_kind(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[_make_call(arguments='{"location": "Dublin"')]
        )
        results = run_checkers(resp, task)
        kinds = [r.failure_kind for r in results if not r.passed]
        assert FailureKind.malformed_json in kinds

    def test_wrong_tool_triggers_wrong_tool_kind(self) -> None:
        task = _make_task(tool_name="get_weather")
        resp = _make_completion(tool_calls=[_make_call("send_email")])
        results = run_checkers(resp, task)
        kinds = [r.failure_kind for r in results if not r.passed]
        assert FailureKind.wrong_tool in kinds

    def test_extra_call_triggers_spurious_call(self) -> None:
        task = _make_task(count=1)
        resp = _make_completion(tool_calls=[_make_call(), _make_call()])
        results = run_checkers(resp, task)
        kinds = [r.failure_kind for r in results if not r.passed]
        assert FailureKind.spurious_call in kinds

    def test_hallucinated_param_triggers_kind(self) -> None:
        task = _make_task()
        resp = _make_completion(
            tool_calls=[
                _make_call(
                    arguments='{"location": "Dublin", "secret_param": "value"}'
                )
            ]
        )
        results = run_checkers(resp, task)
        kinds = [r.failure_kind for r in results if not r.passed]
        assert FailureKind.hallucinated_param in kinds


# ---------------------------------------------------------------------------
# taxonomy — classify_results + primary_kind + severity
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_classify_empty_results_returns_empty(self) -> None:
        results: list[CheckResult] = [
            CheckResult(name="no_call", passed=True),
            CheckResult(name="schema", passed=True),
        ]
        assert classify_results(results) == []

    def test_classify_single_failure(self) -> None:
        results = [
            CheckResult(
                name="no_call",
                passed=False,
                failure_kind=FailureKind.no_call,
            )
        ]
        kinds = classify_results(results)
        assert FailureKind.no_call in kinds

    def test_classify_priority_order(self) -> None:
        """no_call should come before malformed_json in priority order."""
        results = [
            CheckResult(
                name="no_call",
                passed=False,
                failure_kind=FailureKind.no_call,
            ),
            CheckResult(
                name="json",
                passed=False,
                failure_kind=FailureKind.malformed_json,
            ),
        ]
        kinds = classify_results(results)
        assert kinds[0] == FailureKind.no_call

    def test_primary_kind_returns_first(self) -> None:
        kinds = [FailureKind.no_call, FailureKind.wrong_tool]
        assert primary_kind(kinds) == FailureKind.no_call

    def test_primary_kind_empty_list(self) -> None:
        assert primary_kind([]) is None

    def test_severity_critical(self) -> None:
        assert severity(FailureKind.no_call) == "critical"
        assert severity(FailureKind.wrong_tool) == "critical"
        assert severity(FailureKind.malformed_json) == "critical"

    def test_severity_major(self) -> None:
        assert severity(FailureKind.schema_violation) == "major"
        assert severity(FailureKind.missing_required) == "major"
        assert severity(FailureKind.truncation) == "major"

    def test_severity_minor(self) -> None:
        assert severity(FailureKind.type_coercion) == "minor"
        assert severity(FailureKind.escaping_error) == "minor"
        assert severity(FailureKind.spurious_call) == "minor"


# ---------------------------------------------------------------------------
# tasks.py — TaskSpec loading
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_minimal_task(self) -> None:
        task = TaskSpec(
            id="t1",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert task.id == "t1"
        assert task.mode.value == "native_toolcall"

    def test_raises_without_messages(self) -> None:
        with pytest.raises(ValueError):
            TaskSpec(id="t1", messages=[])

    def test_tool_by_name_found(self) -> None:
        task = _make_task()
        tool = task.tool_by_name("get_weather")
        assert tool is not None
        assert tool["function"]["name"] == "get_weather"

    def test_tool_by_name_not_found(self) -> None:
        task = _make_task()
        assert task.tool_by_name("nonexistent") is None

    def test_filter_tasks_empty_tags_returns_all(self) -> None:
        from callcheck.tasks import filter_tasks

        tasks = [_make_task(), _make_task()]
        tasks[0] = TaskSpec(
            id="a", messages=[{"role": "user", "content": "x"}], tags=["smoke"]
        )
        tasks[1] = TaskSpec(
            id="b", messages=[{"role": "user", "content": "x"}], tags=["integration"]
        )
        assert len(filter_tasks(tasks, [])) == 2
        assert len(filter_tasks(tasks, None)) == 2

    def test_filter_tasks_by_tag(self) -> None:
        from callcheck.tasks import filter_tasks

        tasks = [
            TaskSpec(id="a", messages=[{"role": "user", "content": "x"}], tags=["smoke"]),
            TaskSpec(id="b", messages=[{"role": "user", "content": "x"}], tags=["integration"]),
        ]
        result = filter_tasks(tasks, ["smoke"])
        assert len(result) == 1
        assert result[0].id == "a"

    def test_load_task_from_yaml(self, tmp_path) -> None:
        from callcheck.tasks import load_task

        yaml_content = """
id: yaml_test
messages:
  - role: user
    content: "What's the weather?"
tools:
  - type: function
    function:
      name: get_weather
      parameters:
        type: object
        properties:
          location:
            type: string
        required: [location]
expect:
  tool_calls:
    count: 1
    name: get_weather
    required_args: [location]
tags:
  - smoke
"""
        yaml_file = tmp_path / "task.yaml"
        yaml_file.write_text(yaml_content)
        task = load_task(yaml_file)
        assert task.id == "yaml_test"
        assert task.expect.tool_calls is not None
        assert task.expect.tool_calls.name == "get_weather"
        assert "smoke" in task.tags

    def test_load_tasks_scans_directory(self, tmp_path) -> None:
        from callcheck.tasks import load_tasks

        for i in range(3):
            (tmp_path / f"task_{i}.yaml").write_text(
                f"id: t{i}\nmessages:\n  - role: user\n    content: test\n"
            )
        tasks = load_tasks(tmp_path)
        assert len(tasks) == 3
        assert {t.id for t in tasks} == {"t0", "t1", "t2"}
