"""Report generation: render evaluation results as Rich tables or JSON.

Takes a list of :class:`~callcheck.runner.RunResult` objects and renders
summaries, per-task drill-downs, and machine-readable JSON exports suitable
for CI artefacts and dashboards.

Renderers
---------
* ``"rich"``  — coloured tables written to stdout via :mod:`rich`.
* ``"json"``  — a structured dict written as pretty-printed JSON.

Typical usage::

    from callcheck.report import ReportConfig, render
    config = ReportConfig(format="rich", top_n_failures=5)
    render(results, config)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from callcheck.runner import RunResult
from callcheck.taxonomy import FailureKind, severity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ReportConfig:
    """Configuration for report rendering.

    Parameters
    ----------
    format:
        ``"rich"`` for console output, ``"json"`` for machine-readable export.
    output_path:
        When set, write the rendered report to this file instead of stdout.
    top_n_failures:
        Number of failure transcript entries to include in the gallery section.
    show_passed:
        Include passing tasks in the detail rows table.
    """

    format: str = "rich"
    output_path: Path | None = None
    top_n_failures: int = 5
    show_passed: bool = False


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate(results: list[RunResult]) -> dict[str, Any]:
    """Compute summary statistics over *results*."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = passed / total if total else 0.0

    failure_counts: Counter[str] = Counter()
    for r in results:
        if r.primary_failure is not None:
            failure_counts[r.primary_failure.value] += 1

    latencies = [lat for r in results for lat in r.latencies_ms]
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    tok_prompt_all = [t for r in results for t in r.tokens_prompt]
    tok_comp_all = [t for r in results for t in r.tokens_completion]
    mean_tok_p = sum(tok_prompt_all) / len(tok_prompt_all) if tok_prompt_all else 0.0
    mean_tok_c = sum(tok_comp_all) / len(tok_comp_all) if tok_comp_all else 0.0

    # Per-tag breakdown
    tag_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        for tag in r.task_tags:
            stats = tag_stats.setdefault(tag, {"pass": 0, "fail": 0})
            if r.passed:
                stats["pass"] += 1
            else:
                stats["fail"] += 1

    # Agreement average
    mean_agreement = (
        sum(r.agreement for r in results) / total if total else 1.0
    )

    model = results[0].model if results else "unknown"
    parser = results[0].parser_backend if results else "unknown"

    return {
        "model": model,
        "parser_backend": parser,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "mean_latency_ms": mean_lat,
        "mean_tokens_prompt": mean_tok_p,
        "mean_tokens_completion": mean_tok_c,
        "mean_agreement": mean_agreement,
        "failure_mix": dict(failure_counts.most_common()),
        "tag_breakdown": tag_stats,
    }


# ---------------------------------------------------------------------------
# Failure transcript gallery
# ---------------------------------------------------------------------------


def _top_failures(results: list[RunResult], n: int) -> list[RunResult]:
    """Return up to *n* failed results, ordered by primary failure severity."""
    _severity_rank = {"critical": 0, "major": 1, "minor": 2, "unknown": 3}

    def _rank(r: RunResult) -> int:
        if r.primary_failure is None:
            return 99
        return _severity_rank.get(severity(r.primary_failure), 3)

    failed = [r for r in results if not r.passed]
    return sorted(failed, key=_rank)[:n]


def _format_failure_transcript(r: RunResult) -> dict[str, Any]:
    """Render one failure as a structured dict for the gallery."""
    last_response = r.raw_responses[-1] if r.raw_responses else {}
    choices = last_response.get("choices") or []
    msg = choices[0].get("message", {}) if choices else {}
    tool_calls = msg.get("tool_calls") or []

    raw_tool_calls = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        raw_tool_calls.append(
            {
                "name": fn.get("name", "<none>"),
                "arguments": fn.get("arguments", ""),
            }
        )

    failed_checks = [
        {
            "check": cr.name,
            "failure_kind": cr.failure_kind.value if cr.failure_kind else None,
            "detail": cr.detail,
        }
        for cr in r.check_results
        if not cr.passed
    ]

    return {
        "task_id": r.task_id,
        "primary_failure": r.primary_failure.value if r.primary_failure else None,
        "all_failures": [k.value for k in r.failure_kinds],
        "failed_checks": failed_checks,
        "tool_calls_emitted": raw_tool_calls,
        "latency_ms": r.mean_latency_ms,
        "tags": r.task_tags,
    }


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------


def _rich_summary_table(agg: dict[str, Any], console: Console) -> None:
    table = Table(
        title=f"[bold]callcheck — Summary[/bold] ({agg['model']} / {agg['parser_backend']})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim", width=28)
    table.add_column("Value", justify="right")

    pass_rate_pct = agg["pass_rate"] * 100
    rate_color = "green" if pass_rate_pct >= 90 else "yellow" if pass_rate_pct >= 70 else "red"
    table.add_row("Total tasks", str(agg["total"]))
    table.add_row("Passed", f"[green]{agg['passed']}[/green]")
    table.add_row("Failed", f"[red]{agg['failed']}[/red]")
    table.add_row("Pass rate", f"[{rate_color}]{pass_rate_pct:.1f}%[/{rate_color}]")
    table.add_row("Mean latency", f"{agg['mean_latency_ms']:.0f} ms")
    table.add_row("Mean prompt tokens", f"{agg['mean_tokens_prompt']:.0f}")
    table.add_row("Mean completion tokens", f"{agg['mean_tokens_completion']:.0f}")
    table.add_row("Mean k-agreement", f"{agg['mean_agreement'] * 100:.1f}%")
    console.print(table)


def _rich_failure_mix_table(agg: dict[str, Any], console: Console) -> None:
    if not agg["failure_mix"]:
        return
    table = Table(title="Failure Mix", box=box.SIMPLE_HEAD, header_style="bold magenta")
    table.add_column("Failure Kind", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Severity")

    _severity_colors = {"critical": "red", "major": "yellow", "minor": "cyan"}
    for kind_str, count in sorted(
        agg["failure_mix"].items(), key=lambda x: -x[1]
    ):
        try:
            kind = FailureKind(kind_str)
            sev = severity(kind)
        except ValueError:
            sev = "unknown"
        color = _severity_colors.get(sev, "white")
        table.add_row(kind_str, str(count), f"[{color}]{sev}[/{color}]")
    console.print(table)


def _rich_detail_table(results: list[RunResult], show_passed: bool, console: Console) -> None:
    visible = results if show_passed else [r for r in results if not r.passed]
    if not visible:
        console.print("[dim]No failures to show.[/dim]")
        return

    table = Table(
        title="Per-task Results" + ("" if show_passed else " (failures only)"),
        box=box.SIMPLE_HEAD,
        header_style="bold white",
    )
    table.add_column("Task ID", style="dim", min_width=30)
    table.add_column("Verdict", justify="center", width=8)
    table.add_column("Primary Failure", width=22)
    table.add_column("Latency ms", justify="right", width=12)
    table.add_column("Agree %", justify="right", width=8)

    for r in visible:
        verdict = Text("PASS", style="green") if r.passed else Text("FAIL", style="red")
        pf = r.primary_failure.value if r.primary_failure else "—"
        table.add_row(
            r.task_id,
            verdict,
            pf,
            f"{r.mean_latency_ms:.0f}",
            f"{r.agreement * 100:.0f}%",
        )
    console.print(table)


def _rich_failure_gallery(results: list[RunResult], n: int, console: Console) -> None:
    failures = _top_failures(results, n)
    if not failures:
        return

    console.rule(f"[bold red]Top {len(failures)} Failure Transcripts[/bold red]")
    for r in failures:
        entry = _format_failure_transcript(r)
        console.print(f"\n[bold]{entry['task_id']}[/bold] — primary: [red]{entry['primary_failure']}[/red]")
        for fc in entry["failed_checks"]:
            console.print(f"  [yellow]{fc['check']}[/yellow]: {fc['detail']}")
        if entry["tool_calls_emitted"]:
            for tc in entry["tool_calls_emitted"]:
                console.print(f"  [dim]tool_call:[/dim] {tc['name']}({tc['arguments'][:120]}...)"
                               if len(tc['arguments']) > 120
                               else f"  [dim]tool_call:[/dim] {tc['name']}({tc['arguments']})")
        else:
            console.print("  [dim](no tool calls emitted)[/dim]")
        console.print(f"  [dim]tags: {', '.join(entry['tags'])} | latency {entry['latency_ms']:.0f} ms[/dim]")


def _render_rich(
    results: list[RunResult],
    config: ReportConfig,
) -> None:
    if config.output_path is not None:
        with config.output_path.open("w", encoding="utf-8") as fh:
            console = Console(file=fh)
            _render_rich_to(results, config, console)
    else:
        console = Console(file=sys.stdout)
        _render_rich_to(results, config, console)


def _render_rich_to(
    results: list[RunResult],
    config: ReportConfig,
    console: Console,
) -> None:
    if not results:
        console.print("[yellow]No results to render.[/yellow]")
        return

    agg = _aggregate(results)
    _rich_summary_table(agg, console)
    _rich_failure_mix_table(agg, console)
    _rich_detail_table(results, config.show_passed, console)
    _rich_failure_gallery(results, config.top_n_failures, console)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


def _render_json(
    results: list[RunResult],
    config: ReportConfig,
) -> None:
    agg = _aggregate(results)
    gallery = [_format_failure_transcript(r) for r in _top_failures(results, config.top_n_failures)]

    payload = {
        "summary": agg,
        "failure_gallery": gallery,
        "tasks": [r.to_dict() for r in results],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if config.output_path is None:
        print(text)
    else:
        config.output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def summary_table(results: list[RunResult]) -> dict[str, Any]:
    """Return the aggregate stats dict (useful for programmatic access).

    This is the same data :func:`render` shows in the summary section.
    """
    return _aggregate(results)


def failure_gallery(results: list[RunResult], n: int = 5) -> list[dict[str, Any]]:
    """Return up to *n* failure transcript dicts, ordered by severity."""
    return [_format_failure_transcript(r) for r in _top_failures(results, n)]


def render(results: list[RunResult], config: ReportConfig | None = None) -> None:
    """Render results to the console or a file.

    Parameters
    ----------
    results:
        List of :class:`~callcheck.runner.RunResult` objects.
    config:
        Rendering configuration.  Defaults to ``ReportConfig()`` (Rich to stdout).
    """
    if config is None:
        config = ReportConfig()

    if config.format == "json":
        _render_json(results, config)
    else:
        _render_rich(results, config)
