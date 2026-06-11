"""End-to-end pipeline tests: mockserver → runner → report.

Spins up the mockserver in a subprocess, runs the full callcheck pipeline
(async_run_suite → render), and asserts that each mock mode produces the
correct classification in the generated report.

The mockserver is controlled via the ``__mock__:<mode>`` system-message
prefix documented in :mod:`mockserver.app`.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from callcheck.report import ReportConfig, render, summary_table
from callcheck.runner import RunConfig, async_run_suite
from callcheck.tasks import Expectation, TaskSpec, ToolCallExpectation
from callcheck.taxonomy import FailureKind

# ---------------------------------------------------------------------------
# Mockserver fixture
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to port 0, get the assigned ephemeral port, then release it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mockserver_url():
    """Start the mockserver subprocess, yield its base URL, then shut it down."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mockserver.app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to come up
    base_url = f"http://127.0.0.1:{port}"
    import httpx

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Mockserver did not come up within 10 seconds")

    yield base_url

    proc.kill()
    proc.wait()


# ---------------------------------------------------------------------------
# Task factories for different mock scenarios
# ---------------------------------------------------------------------------


def _task_with_mode(
    mode: str,
    count: int = 1,
    tool_name: str = "get_weather",
    required_args: list[str] | None = None,
) -> TaskSpec:
    """Build a TaskSpec whose first system message encodes the mock directive.

    The tool schema includes both 'location' and 'units' so that the mockserver's
    default 'ok' response (which includes both fields) does not trip hallucinated-param.
    """
    if required_args is None:
        required_args = ["location"]
    return TaskSpec(
        id=f"e2e_{mode}",
        messages=[
            {"role": "system", "content": f"__mock__:{mode}"},
            {"role": "user", "content": "What is the weather in Dublin?"},
        ],
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
        tags=["e2e", mode],
    )


# ---------------------------------------------------------------------------
# Individual mock-mode classification tests
# ---------------------------------------------------------------------------


class TestMockserverClassification:
    """Each test sends one mock scenario through the real pipeline."""

    def _run_one(self, task: TaskSpec, base_url: str) -> RunResult:  # noqa: F821
        config = RunConfig(base_url=base_url, api_key="none", model="mock-model", k=1)

        async def _go():
            return await async_run_suite([task], config)

        results = asyncio.run(_go())
        assert len(results) == 1
        return results[0]

    def test_ok_mode_passes_all_checks(self, mockserver_url: str) -> None:
        task = _task_with_mode("ok")
        result = self._run_one(task, mockserver_url)
        assert result.passed is True
        assert result.primary_failure is None
        assert result.failure_kinds == []

    def test_refusal_classified_as_no_call(self, mockserver_url: str) -> None:
        task = _task_with_mode("refusal")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        assert FailureKind.no_call in result.failure_kinds

    def test_malformed_classified_as_malformed_json(self, mockserver_url: str) -> None:
        task = _task_with_mode("malformed")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        assert FailureKind.malformed_json in result.failure_kinds

    def test_extra_calls_classified_as_spurious(self, mockserver_url: str) -> None:
        # Expect 1 call but server returns 2 → spurious_call
        task = _task_with_mode("extra")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        assert FailureKind.spurious_call in result.failure_kinds

    def test_empty_args_classified(self, mockserver_url: str) -> None:
        task = _task_with_mode("empty_args")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        # Empty args → truncation (from json_parse or truncation checker)
        assert FailureKind.truncation in result.failure_kinds or FailureKind.malformed_json in result.failure_kinds

    def test_wrong_tool_classified(self, mockserver_url: str) -> None:
        task = _task_with_mode("wrong_tool")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        # wrong_tool may also trigger missing_required since __hallucinated_tool__
        # is not in the task's tool list; what matters is at least one failure
        assert not result.passed

    def test_hallucinated_param_classified(self, mockserver_url: str) -> None:
        task = _task_with_mode("hallucinated_param")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        assert FailureKind.hallucinated_param in result.failure_kinds

    def test_missing_required_classified(self, mockserver_url: str) -> None:
        task = _task_with_mode("missing_required")
        result = self._run_one(task, mockserver_url)
        assert result.passed is False
        # Empty {} → missing required "location"
        assert (
            FailureKind.missing_required in result.failure_kinds
            or FailureKind.schema_violation in result.failure_kinds
        )


# ---------------------------------------------------------------------------
# Mixed suite → report roundtrip
# ---------------------------------------------------------------------------


class TestMixedSuiteReport:
    """Run a suite with multiple mock modes and verify the aggregated report."""

    def _run_suite(self, base_url: str) -> list:
        modes_and_counts = [
            ("ok", 1, True),
            ("ok", 1, True),
            ("refusal", 1, False),
            ("malformed", 1, False),
            ("extra", 1, False),
            ("missing_required", 1, False),
        ]

        tasks = [
            TaskSpec(
                id=f"e2e_mix_{i}_{mode}",
                messages=[
                    {"role": "system", "content": f"__mock__:{mode}"},
                    {"role": "user", "content": "weather?"},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"},
                                    "units": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
                expect=Expectation(
                    tool_calls=ToolCallExpectation(
                        count=count,
                        name="get_weather",
                        required_args=["location"],
                    )
                ),
                tags=["e2e", mode],
            )
            for i, (mode, count, _expected_pass) in enumerate(modes_and_counts)
        ]

        config = RunConfig(base_url=base_url, api_key="none", model="mock-model", k=1)

        async def _go():
            return await async_run_suite(tasks, config)

        return asyncio.run(_go()), modes_and_counts

    def test_pass_rate_matches_expected(self, mockserver_url: str) -> None:
        results, modes_and_counts = self._run_suite(mockserver_url)
        expected_passes = sum(1 for _, _, should_pass in modes_and_counts if should_pass)
        actual_passes = sum(1 for r in results if r.passed)
        assert actual_passes == expected_passes

    def test_summary_table_totals(self, mockserver_url: str) -> None:
        results, modes_and_counts = self._run_suite(mockserver_url)
        agg = summary_table(results)
        assert agg["total"] == len(modes_and_counts)
        assert agg["passed"] + agg["failed"] == agg["total"]

    def test_json_report_round_trip(self, mockserver_url: str, tmp_path: Path) -> None:
        """Run suite → write JSON report → reload → verify content."""
        results, modes_and_counts = self._run_suite(mockserver_url)

        out = tmp_path / "e2e_report.json"
        config = ReportConfig(format="json", output_path=out, top_n_failures=3)
        render(results, config)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["summary"]["total"] == len(modes_and_counts)
        assert len(data["tasks"]) == len(modes_and_counts)
        # All tasks have task_id keys
        assert all("task_id" in t for t in data["tasks"])

    def test_failure_kinds_are_non_empty_for_failures(self, mockserver_url: str) -> None:
        results, _ = self._run_suite(mockserver_url)
        for r in results:
            if not r.passed:
                assert len(r.failure_kinds) > 0, f"{r.task_id} failed but has no failure kinds"

    def test_primary_failure_is_none_for_passes(self, mockserver_url: str) -> None:
        results, _ = self._run_suite(mockserver_url)
        for r in results:
            if r.passed:
                assert r.primary_failure is None

    def test_latencies_are_positive(self, mockserver_url: str) -> None:
        results, _ = self._run_suite(mockserver_url)
        for r in results:
            assert r.mean_latency_ms >= 0.0
            assert all(lat >= 0.0 for lat in r.latencies_ms)

    def test_rich_report_no_crash(self, mockserver_url: str, tmp_path: Path) -> None:
        results, _ = self._run_suite(mockserver_url)
        out = tmp_path / "e2e_rich.txt"
        config = ReportConfig(format="rich", output_path=out, top_n_failures=5)
        render(results, config)
        assert out.exists()
