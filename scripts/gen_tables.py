"""gen_tables.py — generate Markdown/HTML comparison tables from callcheck JSON results.

Reads one or more JSON result files produced by 'callcheck report --format json'
and emits a side-by-side Markdown table comparing pass rates across models or runs.

TODO:
  - parse CLI args: input glob pattern, output file, format (md/html)
  - load_results(path) -> list[dict]: parse JSON lines result files
  - pivot_table(results) -> pd.DataFrame: task_id x model -> pass/fail
  - render_markdown(df) -> str
  - render_html(df) -> str: include colour-coded cells (green/red/amber)
  - main(): glue together, write output file

Usage (planned):
  python scripts/gen_tables.py results/*.json --output report.md
"""

if __name__ == "__main__":
    raise NotImplementedError("gen_tables.py is not yet implemented.")
