"""Report generation: render evaluation results as Rich tables, JSON, or HTML.

Takes a list of RunResult objects from runner.py and renders human-readable
summaries, per-task drill-downs, and machine-readable JSON exports.

TODO:
  - ReportConfig dataclass (format: Literal["rich", "json", "html"], output_path)
  - summary_table(results) -> rich.Table: pass/fail counts by tag/category
  - detail_rows(results) -> rich.Table: per-task verdict + failure detail
  - export_json(results, path): write JSON lines for downstream processing
  - render(results, config): dispatch to the right renderer
"""
