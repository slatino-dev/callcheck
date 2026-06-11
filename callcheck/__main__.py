"""CLI entry point for callcheck.

TODO: wire up typer app with subcommands:
  callcheck run   -- execute a task suite against a model endpoint
  callcheck report -- render results from a JSON result file
  callcheck list  -- list available tasks (with tag filter)
"""

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("callcheck requires 'typer'. Install with: pip install typer") from exc

app = typer.Typer(name="callcheck", help="Tool-calling conformance tester for vLLM-served models.")


@app.command()
def run() -> None:
    """Run a conformance task suite against a model endpoint (not yet implemented)."""
    raise NotImplementedError("'callcheck run' is not yet implemented.")


if __name__ == "__main__":
    app()
