"""CLI entry point for callcheck.

Sub-commands
------------
callcheck run    — run a task suite against a model endpoint
callcheck report — render results from a JSON/JSONL checkpoint file
callcheck list   — list available tasks with optional tag filtering

Examples::

    # Run the full suite against a local vLLM instance
    callcheck run --tasks tasks/ --model qwen2.5-7b --base-url http://127.0.0.1:8000

    # Run only smoke-tagged tasks, 3 repeats, async with 4 concurrent workers
    callcheck run --tasks tasks/ --tags smoke --k 3 --concurrency 4

    # Show results from a prior checkpoint
    callcheck report --input results/qwen.jsonl

    # List tasks matching a tag
    callcheck list --tasks tasks/ --tags parallel
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "callcheck requires 'typer' and 'rich'. Install with: pip install typer rich"
    ) from exc

from callcheck.report import ReportConfig, render
from callcheck.runner import RunConfig, RunResult, async_run_suite, load_checkpoint_results
from callcheck.tasks import filter_tasks, load_tasks
from callcheck.taxonomy import FailureKind

app = typer.Typer(
    name="callcheck",
    help="Tool-calling conformance tester for OpenAI-compatible model endpoints.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# callcheck run
# ---------------------------------------------------------------------------


@app.command()
def run(
    tasks_dir: Annotated[
        Path,
        typer.Option("--tasks", "-t", help="Directory containing task YAML files.", show_default=False),
    ] = Path("tasks"),
    model: Annotated[str, typer.Option("--model", "-m", help="Model identifier.")] = "default",
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="API root URL (overrides OPENAI_BASE_URL env var)."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Bearer token (overrides OPENAI_API_KEY env var)."),
    ] = None,
    parser_backend: Annotated[
        str,
        typer.Option("--parser-backend", help="Parser backend label (metadata only)."),
    ] = "default",
    k: Annotated[int, typer.Option("--k", help="Number of repeats per task.")] = 1,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j", help="Async concurrency.")
    ] = 4,
    timeout: Annotated[float, typer.Option("--timeout", help="Request timeout in seconds.")] = 60.0,
    tags: Annotated[
        str | None,
        typer.Option("--tags", help="Comma-separated tag filter (AND semantics)."),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="JSONL checkpoint file (resumable run)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON report to this file."),
    ] = None,
    report_format: Annotated[
        str, typer.Option("--format", "-f", help="Report format: rich or json.")
    ] = "rich",
    top_n: Annotated[
        int, typer.Option("--top-n", help="Number of failure transcripts in gallery.")
    ] = 5,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 0.0,
) -> None:
    """Run a conformance task suite against a model endpoint."""
    if not tasks_dir.is_dir():
        console.print(f"[red]Tasks directory not found: {tasks_dir}[/red]")
        raise typer.Exit(code=1)

    all_tasks = load_tasks(tasks_dir)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    filtered = filter_tasks(all_tasks, tag_list)

    if not filtered:
        console.print("[yellow]No tasks matched the given filters.[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"[cyan]Running {len(filtered)} task(s)"
        f" (model={model}, k={k}, concurrency={concurrency})[/cyan]"
    )

    config = RunConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        parser_backend=parser_backend,
        k=k,
        concurrency=concurrency,
        timeout=timeout,
        max_retries=2,
        checkpoint_path=checkpoint,
        tags=tag_list,
        temperature=temperature,
    )

    def _progress(result: RunResult, done: int, total: int) -> None:
        icon = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        pf = f" ({result.primary_failure.value})" if result.primary_failure else ""
        console.print(f"  [{done:>{len(str(total))}}/{total}] {icon} {result.task_id}{pf}")

    results: list[RunResult] = asyncio.run(
        async_run_suite(filtered, config, progress=_progress)
    )

    report_cfg = ReportConfig(
        format=report_format,
        output_path=output,
        top_n_failures=top_n,
        show_passed=False,
    )
    render(results, report_cfg)

    pass_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    console.print(
        f"\n[bold]Done:[/bold] {pass_count}/{total_count} passed "
        f"({pass_count / total_count * 100:.1f}%)"
    )

    # Exit non-zero if any failures so CI gates can catch it
    if pass_count < total_count:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# callcheck report
# ---------------------------------------------------------------------------


@app.command()
def report(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="JSONL checkpoint or JSON result file.", show_default=False),
    ],
    report_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format: rich or json.")
    ] = "rich",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write rendered report to this file."),
    ] = None,
    top_n: Annotated[
        int, typer.Option("--top-n", help="Number of failure transcripts.")
    ] = 5,
) -> None:
    """Render results from a prior checkpoint file."""
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(code=1)

    rows = load_checkpoint_results(input_path)
    if not rows:
        # Try as pretty-printed JSON with a top-level "tasks" key
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
            rows = data.get("tasks", [])
        except (json.JSONDecodeError, ValueError):
            pass

    if not rows:
        console.print("[yellow]No results found in the input file.[/yellow]")
        raise typer.Exit(code=0)

    # Reconstruct lightweight RunResult objects for rendering
    results: list[RunResult] = []
    for row in rows:
        # Build a stub RunResult from the serialised dict
        fk_list = [
            FailureKind(k) for k in (row.get("failure_kinds") or [])
            if k in {fk.value for fk in FailureKind}
        ]
        pf_str = row.get("primary_failure")
        pf = FailureKind(pf_str) if pf_str and pf_str in {fk.value for fk in FailureKind} else None
        results.append(
            RunResult(
                task_id=row.get("task_id", "unknown"),
                passed=bool(row.get("passed", False)),
                agreement=float(row.get("agreement", 1.0)),
                check_results=[],
                failure_kinds=fk_list,
                primary_failure=pf,
                raw_responses=[],
                latencies_ms=row.get("latencies_ms") or [],
                tokens_prompt=row.get("tokens_prompt") or [],
                tokens_completion=row.get("tokens_completion") or [],
                model=row.get("model", "unknown"),
                parser_backend=row.get("parser_backend", "unknown"),
                task_tags=row.get("task_tags") or [],
            )
        )


    report_cfg = ReportConfig(
        format=report_format,
        output_path=output,
        top_n_failures=top_n,
        show_passed=False,
    )
    render(results, report_cfg)


# ---------------------------------------------------------------------------
# callcheck list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_tasks(
    tasks_dir: Annotated[
        Path,
        typer.Option("--tasks", "-t", help="Directory containing task YAML files."),
    ] = Path("tasks"),
    tags: Annotated[
        str | None,
        typer.Option("--tags", help="Comma-separated tag filter (AND semantics)."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show descriptions.")] = False,
) -> None:
    """List available tasks with optional tag filtering."""
    if not tasks_dir.is_dir():
        console.print(f"[red]Tasks directory not found: {tasks_dir}[/red]")
        raise typer.Exit(code=1)

    all_tasks = load_tasks(tasks_dir)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    filtered = filter_tasks(all_tasks, tag_list)

    table = Table(title=f"Tasks ({len(filtered)} / {len(all_tasks)} shown)", show_header=True)
    table.add_column("ID", style="bold cyan", min_width=35)
    table.add_column("Tags", style="dim")
    table.add_column("Mode", width=18)
    if verbose:
        table.add_column("Description")

    for task in filtered:
        row = [
            task.id,
            ", ".join(task.tags),
            task.mode.value,
        ]
        if verbose:
            row.append(task.description[:80] or "—")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    app()
