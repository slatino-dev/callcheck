"""Unit tests for report.py.

Verifies summary aggregation, failure sorting, gallery generation, and
both Rich and JSON renderers without a real terminal (uses StringIO).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from callcheck.report import (
    ReportConfig,
    _aggregate,
    _top_failures,
    failure_gallery,
    render,
    summary_table,
)
from callcheck.runner import RunResult
from callcheck.taxonomy import FailureKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    task_id: str,
    passed: bool,
    primary: FailureKind | None = None,
    latencies: list[float] | None = None,
    tags: list[str] | None = None,
    all_kinds: list[FailureKind] | None = None,
    model: str = "mock",
    parser: str = "default",
) -> RunResult:
    fk = [primary] if primary else []
    if all_kinds:
        fk = all_kinds
    return RunResult(
        task_id=task_id,
        passed=passed,
        agreement=1.0 if passed else 0.7,
        check_results=[],
        failure_kinds=fk,
        primary_failure=primary,
        raw_responses=[{}],
        latencies_ms=latencies or [100.0],
        tokens_prompt=[20],
        tokens_completion=[10],
        model=model,
        parser_backend=parser,
        task_tags=tags or [],
    )


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_all_pass(self) -> None:
        results = [_make_result(f"t{i}", passed=True) for i in range(5)]
        agg = _aggregate(results)
        assert agg["passed"] == 5
        assert agg["failed"] == 0
        assert agg["pass_rate"] == pytest.approx(1.0)

    def test_all_fail(self) -> None:
        results = [_make_result(f"t{i}", passed=False, primary=FailureKind.no_call) for i in range(3)]
        agg = _aggregate(results)
        assert agg["passed"] == 0
        assert agg["failed"] == 3
        assert agg["pass_rate"] == pytest.approx(0.0)

    def test_mixed_pass_rate(self) -> None:
        results = [
            _make_result("t1", passed=True),
            _make_result("t2", passed=True),
            _make_result("t3", passed=False, primary=FailureKind.malformed_json),
            _make_result("t4", passed=False, primary=FailureKind.no_call),
        ]
        agg = _aggregate(results)
        assert agg["pass_rate"] == pytest.approx(0.5)

    def test_failure_mix_counts(self) -> None:
        results = [
            _make_result("t1", passed=False, primary=FailureKind.no_call),
            _make_result("t2", passed=False, primary=FailureKind.no_call),
            _make_result("t3", passed=False, primary=FailureKind.malformed_json),
        ]
        agg = _aggregate(results)
        assert agg["failure_mix"]["no_call"] == 2
        assert agg["failure_mix"]["malformed_json"] == 1

    def test_mean_latency(self) -> None:
        results = [
            _make_result("t1", passed=True, latencies=[100.0]),
            _make_result("t2", passed=True, latencies=[200.0]),
        ]
        agg = _aggregate(results)
        assert agg["mean_latency_ms"] == pytest.approx(150.0)

    def test_tag_breakdown(self) -> None:
        results = [
            _make_result("t1", passed=True, tags=["smoke", "single"]),
            _make_result("t2", passed=False, primary=FailureKind.no_call, tags=["smoke"]),
        ]
        agg = _aggregate(results)
        assert agg["tag_breakdown"]["smoke"]["pass"] == 1
        assert agg["tag_breakdown"]["smoke"]["fail"] == 1
        assert agg["tag_breakdown"]["single"]["pass"] == 1

    def test_empty_results_no_crash(self) -> None:
        agg = _aggregate([])
        assert agg["total"] == 0
        assert agg["pass_rate"] == pytest.approx(0.0)

    def test_mean_agreement(self) -> None:
        results = [
            _make_result("t1", passed=True),  # agreement=1.0
            _make_result("t2", passed=False, primary=FailureKind.no_call),  # agreement=0.7
        ]
        agg = _aggregate(results)
        assert agg["mean_agreement"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# _top_failures / failure_gallery
# ---------------------------------------------------------------------------


class TestTopFailures:
    def test_returns_only_failures(self) -> None:
        results = [
            _make_result("t1", passed=True),
            _make_result("t2", passed=False, primary=FailureKind.no_call),
        ]
        top = _top_failures(results, 5)
        assert len(top) == 1
        assert top[0].task_id == "t2"

    def test_respects_n_limit(self) -> None:
        results = [
            _make_result(f"t{i}", passed=False, primary=FailureKind.no_call)
            for i in range(10)
        ]
        top = _top_failures(results, 3)
        assert len(top) == 3

    def test_critical_before_minor(self) -> None:
        results = [
            _make_result("t_minor", passed=False, primary=FailureKind.spurious_call),
            _make_result("t_critical", passed=False, primary=FailureKind.no_call),
        ]
        top = _top_failures(results, 5)
        assert top[0].task_id == "t_critical"

    def test_gallery_structure(self) -> None:
        results = [
            _make_result("fail1", passed=False, primary=FailureKind.malformed_json),
        ]
        gallery = failure_gallery(results, n=1)
        assert len(gallery) == 1
        entry = gallery[0]
        assert entry["task_id"] == "fail1"
        assert entry["primary_failure"] == "malformed_json"
        assert "failed_checks" in entry
        assert "tool_calls_emitted" in entry

    def test_gallery_empty_when_all_pass(self) -> None:
        results = [_make_result(f"t{i}", passed=True) for i in range(3)]
        assert failure_gallery(results, n=5) == []


# ---------------------------------------------------------------------------
# summary_table (public API)
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_returns_dict_with_expected_keys(self) -> None:
        results = [_make_result("t1", passed=True), _make_result("t2", passed=False)]
        agg = summary_table(results)
        for key in ("total", "passed", "failed", "pass_rate", "failure_mix", "tag_breakdown"):
            assert key in agg, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# render — JSON format
# ---------------------------------------------------------------------------


class TestRenderJson:
    def test_json_output_structure(self, tmp_path: Path) -> None:
        results = [
            _make_result("t1", passed=True),
            _make_result("t2", passed=False, primary=FailureKind.no_call),
        ]
        out = tmp_path / "report.json"
        config = ReportConfig(format="json", output_path=out, top_n_failures=5)
        render(results, config)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "summary" in data
        assert "tasks" in data
        assert "failure_gallery" in data
        assert data["summary"]["total"] == 2

    def test_json_tasks_list_matches_results(self, tmp_path: Path) -> None:
        results = [_make_result(f"t{i}", passed=(i % 2 == 0)) for i in range(4)]
        out = tmp_path / "report.json"
        config = ReportConfig(format="json", output_path=out)
        render(results, config)

        data = json.loads(out.read_text())
        assert len(data["tasks"]) == 4
        ids = {t["task_id"] for t in data["tasks"]}
        assert ids == {"t0", "t1", "t2", "t3"}

    def test_json_gallery_capped_at_top_n(self, tmp_path: Path) -> None:
        results = [
            _make_result(f"t{i}", passed=False, primary=FailureKind.no_call)
            for i in range(10)
        ]
        out = tmp_path / "report.json"
        config = ReportConfig(format="json", output_path=out, top_n_failures=3)
        render(results, config)

        data = json.loads(out.read_text())
        assert len(data["failure_gallery"]) == 3

    def test_json_printed_to_stdout_when_no_output_path(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        results = [_make_result("t1", passed=True)]
        config = ReportConfig(format="json")
        render(results, config)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total"] == 1


# ---------------------------------------------------------------------------
# render — Rich format (smoke, no crash)
# ---------------------------------------------------------------------------


class TestRenderRich:
    def test_render_rich_no_crash(self, tmp_path: Path) -> None:
        """Rich output to a file must not raise."""
        results = [
            _make_result("t1", passed=True, tags=["smoke"]),
            _make_result("t2", passed=False, primary=FailureKind.wrong_tool, tags=["parallel"]),
        ]
        out = tmp_path / "report.txt"
        config = ReportConfig(format="rich", output_path=out, top_n_failures=2)
        render(results, config)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "t2" in content

    def test_render_empty_results_no_crash(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.txt"
        config = ReportConfig(format="rich", output_path=out)
        render([], config)
        assert out.exists()

    def test_render_default_config(self, tmp_path: Path) -> None:
        """render() with no config arg defaults to Rich stdout."""
        results = [_make_result("t1", passed=True)]
        # Should not raise even without config
        out = tmp_path / "default.txt"
        config = ReportConfig(format="rich", output_path=out)
        render(results, config)
