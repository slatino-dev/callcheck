"""Unit tests for runner.py.

Tests the RunConfig, RunResult, run_task, run_suite, async_run_suite, and
checkpoint machinery without a real network connection by patching the client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from callcheck.runner import (
    RunConfig,
    RunResult,
    _append_checkpoint,
    _load_checkpoint,
    async_run_suite,
    load_checkpoint_results,
    run_suite,
    run_task,
)
from callcheck.tasks import Expectation, TaskSpec, ToolCallExpectation
from callcheck.taxonomy import FailureKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "t1",
    tool_name: str = "get_weather",
    count: int = 1,
    required_args: list[str] | None = None,
) -> TaskSpec:
    if required_args is None:
        required_args = ["location"]
    return TaskSpec(
        id=task_id,
        messages=[{"role": "user", "content": "What is the weather in Dublin?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": required_args,
                    },
                },
            }
        ],
        expect=Expectation(
            tool_calls=ToolCallExpectation(
                count=count,
                name=tool_name,
                required_args=required_args,
            )
        ),
        tags=["smoke", "single_tool"],
    )


def _make_good_response(tool_name: str = "get_weather") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"location": "Dublin", "units": "celsius"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def _make_refusal_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I can't help.",
                    "tool_calls": None,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_default_model(self) -> None:
        config = RunConfig()
        assert config.model == "default"
        assert config.k == 1
        assert config.concurrency == 4
        assert config.temperature == 0.0

    def test_custom_config(self) -> None:
        config = RunConfig(model="qwen3-8b", k=3, concurrency=2, parser_backend="hermes")
        assert config.model == "qwen3-8b"
        assert config.k == 3
        assert config.parser_backend == "hermes"


# ---------------------------------------------------------------------------
# RunResult aggregates
# ---------------------------------------------------------------------------


class TestRunResult:
    def _make_result(self, latencies: list[float], toks_p: list[int], toks_c: list[int]) -> RunResult:
        return RunResult(
            task_id="t1",
            passed=True,
            agreement=1.0,
            check_results=[],
            failure_kinds=[],
            primary_failure=None,
            raw_responses=[],
            latencies_ms=latencies,
            tokens_prompt=toks_p,
            tokens_completion=toks_c,
            model="mock",
            parser_backend="default",
            task_tags=["smoke"],
        )

    def test_mean_latency_single(self) -> None:
        r = self._make_result([150.0], [10], [5])
        assert r.mean_latency_ms == pytest.approx(150.0)

    def test_mean_latency_multiple(self) -> None:
        r = self._make_result([100.0, 200.0, 300.0], [10, 10, 10], [5, 5, 5])
        assert r.mean_latency_ms == pytest.approx(200.0)

    def test_mean_latency_empty(self) -> None:
        r = self._make_result([], [], [])
        assert r.mean_latency_ms == 0.0

    def test_mean_tokens_prompt(self) -> None:
        r = self._make_result([100.0], [20, 30], [10, 10])
        assert r.mean_tokens_prompt == pytest.approx(25.0)

    def test_mean_tokens_completion(self) -> None:
        r = self._make_result([100.0], [20], [5, 15])
        assert r.mean_tokens_completion == pytest.approx(10.0)

    def test_to_dict_keys(self) -> None:
        r = self._make_result([100.0], [10], [5])
        d = r.to_dict()
        assert "task_id" in d
        assert "passed" in d
        assert "failure_kinds" in d
        assert "latencies_ms" in d
        assert "model" in d
        # raw_responses and check_results are intentionally excluded
        assert "raw_responses" not in d
        assert "check_results" not in d

    def test_to_dict_failure_kinds_serialised(self) -> None:
        r = RunResult(
            task_id="t1",
            passed=False,
            agreement=0.5,
            check_results=[],
            failure_kinds=[FailureKind.no_call, FailureKind.wrong_tool],
            primary_failure=FailureKind.no_call,
            raw_responses=[],
            latencies_ms=[100.0],
            tokens_prompt=[10],
            tokens_completion=[5],
            model="m",
            parser_backend="p",
        )
        d = r.to_dict()
        assert d["failure_kinds"] == ["no_call", "wrong_tool"]
        assert d["primary_failure"] == "no_call"


# ---------------------------------------------------------------------------
# run_task (synchronous, k=1 and k>1)
# ---------------------------------------------------------------------------


class TestRunTask:
    def test_single_pass(self) -> None:
        task = _make_task()
        config = RunConfig(k=1)
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_good_response()

        result = run_task(task, mock_client, config)

        assert result.passed is True
        assert result.task_id == "t1"
        assert result.agreement == 1.0
        assert result.failure_kinds == []
        assert result.primary_failure is None
        assert len(result.raw_responses) == 1
        assert len(result.latencies_ms) == 1
        assert result.tokens_prompt == [20]
        assert result.tokens_completion == [10]

    def test_single_fail_refusal(self) -> None:
        task = _make_task()
        config = RunConfig(k=1)
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_refusal_response()

        result = run_task(task, mock_client, config)

        assert result.passed is False
        assert result.primary_failure == FailureKind.no_call

    def test_k3_all_pass(self) -> None:
        task = _make_task()
        config = RunConfig(k=3)
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_good_response()

        result = run_task(task, mock_client, config)

        assert result.passed is True
        assert result.agreement == 1.0
        assert len(result.raw_responses) == 3
        assert len(result.latencies_ms) == 3
        assert mock_client.complete.call_count == 3

    def test_k3_all_fail_agreement_is_one(self) -> None:
        """All 3 fail → majority is fail → agreement=1.0."""
        task = _make_task()
        config = RunConfig(k=3)
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_refusal_response()

        result = run_task(task, mock_client, config)

        assert result.passed is False
        assert result.agreement == pytest.approx(1.0)

    def test_k2_split_one_pass_one_fail(self) -> None:
        """k=2, 1 pass + 1 fail → passed=False (strict), agreement=0.5."""
        task = _make_task()
        config = RunConfig(k=2)
        mock_client = MagicMock()
        responses = [_make_good_response(), _make_refusal_response()]
        mock_client.complete.side_effect = responses

        result = run_task(task, mock_client, config)

        assert result.passed is False
        assert result.agreement == pytest.approx(0.5)

    def test_task_tags_propagated(self) -> None:
        task = _make_task()
        config = RunConfig()
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_good_response()

        result = run_task(task, mock_client, config)

        assert "smoke" in result.task_tags
        assert "single_tool" in result.task_tags

    def test_model_and_parser_propagated(self) -> None:
        task = _make_task()
        config = RunConfig(model="hermes-3", parser_backend="hermes-2")
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_good_response()

        result = run_task(task, mock_client, config)

        assert result.model == "hermes-3"
        assert result.parser_backend == "hermes-2"

    def test_missing_usage_fields_default_to_zero(self) -> None:
        task = _make_task()
        config = RunConfig()
        mock_client = MagicMock()
        response = _make_good_response()
        response["usage"] = {}
        mock_client.complete.return_value = response

        result = run_task(task, mock_client, config)

        assert result.tokens_prompt == [0]
        assert result.tokens_completion == [0]


# ---------------------------------------------------------------------------
# run_suite
# ---------------------------------------------------------------------------


class TestRunSuite:
    def test_returns_result_per_task(self) -> None:
        tasks = [_make_task("t1"), _make_task("t2")]
        config = RunConfig()

        with patch("callcheck.runner.ModelClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = _make_good_response()
            results = run_suite(tasks, config)

        assert len(results) == 2
        assert {r.task_id for r in results} == {"t1", "t2"}

    def test_progress_callback_fires_for_each_task(self) -> None:
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        config = RunConfig()
        fired: list[tuple[str, int, int]] = []

        def _cb(result: RunResult, done: int, total: int) -> None:
            fired.append((result.task_id, done, total))

        with patch("callcheck.runner.ModelClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = _make_good_response()
            run_suite(tasks, config, progress=_cb)

        assert len(fired) == 3
        assert all(total == 3 for _, _, total in fired)

    def test_checkpoint_skips_already_done(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.jsonl"
        # Pre-populate checkpoint with t1
        existing = RunResult(
            task_id="t1",
            passed=True,
            agreement=1.0,
            check_results=[],
            failure_kinds=[],
            primary_failure=None,
            raw_responses=[],
            latencies_ms=[50.0],
            tokens_prompt=[10],
            tokens_completion=[5],
            model="m",
            parser_backend="p",
        )
        _append_checkpoint(ckpt, existing)

        tasks = [_make_task("t1"), _make_task("t2")]
        config = RunConfig(checkpoint_path=ckpt)

        with patch("callcheck.runner.ModelClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = _make_good_response()
            results = run_suite(tasks, config)

        # Only t2 should be run (t1 was in checkpoint)
        assert len(results) == 1
        assert results[0].task_id == "t2"

    def test_checkpoint_appends_new_results(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.jsonl"
        tasks = [_make_task("t1")]
        config = RunConfig(checkpoint_path=ckpt)

        with patch("callcheck.runner.ModelClient") as MockClient:
            instance = MockClient.return_value
            instance.complete.return_value = _make_good_response()
            run_suite(tasks, config)

        rows = load_checkpoint_results(ckpt)
        assert len(rows) == 1
        assert rows[0]["task_id"] == "t1"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    def test_load_checkpoint_empty_file(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.jsonl"
        ckpt.write_text("")
        assert _load_checkpoint(ckpt) == set()

    def test_load_checkpoint_missing_file(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "nonexistent.jsonl"
        assert _load_checkpoint(ckpt) == set()

    def test_load_checkpoint_returns_ids(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.jsonl"
        ckpt.write_text(
            json.dumps({"task_id": "t1", "passed": True}) + "\n"
            + json.dumps({"task_id": "t2", "passed": False}) + "\n"
        )
        seen = _load_checkpoint(ckpt)
        assert seen == {"t1", "t2"}

    def test_load_checkpoint_skips_malformed_lines(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt.jsonl"
        ckpt.write_text('{"task_id": "t1"}\nnot-json\n{"task_id": "t3"}\n')
        seen = _load_checkpoint(ckpt)
        assert "t1" in seen
        assert "t3" in seen

    def test_append_creates_file(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "sub" / "ckpt.jsonl"
        result = RunResult(
            task_id="t_new",
            passed=True,
            agreement=1.0,
            check_results=[],
            failure_kinds=[],
            primary_failure=None,
            raw_responses=[],
            latencies_ms=[99.0],
            tokens_prompt=[5],
            tokens_completion=[3],
            model="m",
            parser_backend="p",
        )
        _append_checkpoint(ckpt, result)
        assert ckpt.exists()
        rows = load_checkpoint_results(ckpt)
        assert rows[0]["task_id"] == "t_new"
        assert rows[0]["latencies_ms"] == [99.0]

    def test_load_checkpoint_results_empty(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "empty.jsonl"
        assert load_checkpoint_results(ckpt) == []


# ---------------------------------------------------------------------------
# async_run_suite
# ---------------------------------------------------------------------------


class TestAsyncRunSuite:
    def test_async_suite_returns_all_results(self) -> None:
        tasks = [_make_task(f"t{i}") for i in range(5)]
        config = RunConfig(concurrency=3)

        async def _run() -> list[RunResult]:
            with patch("callcheck.runner.AsyncModelClient") as MockClient:
                instance = MockClient.return_value

                async def _acomplete(*args, **kwargs):
                    return _make_good_response()

                instance.acomplete = _acomplete
                return await async_run_suite(tasks, config)

        results = asyncio.run(_run())
        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_async_suite_order_preserved(self) -> None:
        """Results must be returned in the original task order."""
        task_ids = [f"task_{i}" for i in range(4)]
        tasks = [_make_task(tid) for tid in task_ids]
        config = RunConfig(concurrency=2)

        async def _run() -> list[RunResult]:
            with patch("callcheck.runner.AsyncModelClient") as MockClient:
                instance = MockClient.return_value

                async def _acomplete(*args, **kwargs):
                    return _make_good_response()

                instance.acomplete = _acomplete
                return await async_run_suite(tasks, config)

        results = asyncio.run(_run())
        assert [r.task_id for r in results] == task_ids

    def test_async_suite_progress_fires(self) -> None:
        tasks = [_make_task(f"t{i}") for i in range(3)]
        config = RunConfig(concurrency=2)
        fired = []

        async def _run() -> list[RunResult]:
            with patch("callcheck.runner.AsyncModelClient") as MockClient:
                instance = MockClient.return_value

                async def _acomplete(*args, **kwargs):
                    return _make_good_response()

                instance.acomplete = _acomplete

                def _cb(result: RunResult, done: int, total: int) -> None:
                    fired.append(done)

                return await async_run_suite(tasks, config, progress=_cb)

        asyncio.run(_run())
        assert len(fired) == 3
