"""Test runner that drives tasks through a ModelClient and collects results.

Orchestrates the full evaluation loop: load tasks → send to model → apply
checkers → aggregate scores → hand off to report.py.  Supports both
synchronous (simple CLI) and async (parallel) execution modes.

TODO:
  - RunConfig dataclass (model, base_url, concurrency, timeout, task_filter)
  - RunResult dataclass (task_id, raw_response, check_results, latency_ms)
  - run_task(task, client, config) -> RunResult
  - run_suite(tasks, config) -> list[RunResult]: sequential fallback
  - async_run_suite(tasks, config) -> list[RunResult]: asyncio parallel
  - progress callback hook for Rich live display
"""
