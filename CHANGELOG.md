# Changelog

All notable changes to callcheck are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-11

### Added

- **10-stage checker pipeline** (`check_no_call` → `check_escaping`) that scores an
  OpenAI-compatible chat-completion response against a typed `TaskSpec`.
- **12-label failure taxonomy** (`FailureKind` StrEnum) with severity tiers and a
  deterministic priority ordering for multi-failure classification.
- **42-task YAML corpus** across five categories: `single`, `schema`, `parallel`,
  `nocall`, `edge`.  Edge cases include RTL text, emoji, embedded-JSON strings, float
  precision, `$ref` / `anyOf` schemas, forced `tool_choice`, and parallel-collapse traps.
- **k-repeat runner** with agreement metric, resumable JSONL checkpointing, bounded-
  concurrency async dispatch (`async_run_suite`), and non-zero exit on failures.
- **Persistent `AsyncModelClient`** reusing one `httpx.AsyncClient` across all requests
  in a suite; supports `async with` for explicit lifecycle management.
- **Rich console report** (summary table, failure-mix table, per-task detail, failure
  transcript gallery) and JSON export suitable for CI artefacts.
- **Deterministic mockserver** (FastAPI + uvicorn) with nine controlled fault modes
  (`ok`, `refusal`, `malformed`, `extra`, `empty_args`, `wrong_tool`,
  `hallucinated_param`, `missing_required`, default).
- **168-test suite**: 68 unit tests over every checker and taxonomy label; runner tests
  with `MagicMock` clients and `tmp_path` checkpoint round-trips; end-to-end tests that
  spawn the mockserver as a subprocess on an ephemeral port and assert specific
  `FailureKind` classification per fault mode.
- CI: ruff (strict select), mypy, pytest + pytest-asyncio across Python 3.11 / 3.12,
  and a secret-scrub gate that runs before lint and test.
- `callcheck run` / `callcheck report` CLI via Typer.

### Fixed

- `check_no_call` now correctly passes when `task.expect.tool_calls.count == 0`:
  a content-only response is the *correct* output for conversational tasks that expect
  no tool call.  Previously, every `nocall/*.yaml` task was unsatisfiable.
- File-handle leak in `report._render_rich`: the output file is now opened with a
  `with` statement so the handle is closed after rendering (relevant on Windows).
- Unreachable branch in `check_call_count` (`expected_count > 1 and actual == 1`
  inside the `actual > expected_count` arm) removed.
- `check_truncation` docstring now matches the implementation: it catches empty /
  `None` arguments, not a short-args heuristic (which was only `pass`).
- `ProgressCallback` type alias comment corrected to `(result, done, total)`.

### Removed

- `spark_matrix_guide()` from `runner.py`: was an agent directive shipped as product
  code.  Matrix orchestration procedure is now documented in the README.
- `scripts/gen_tables.py`: dead `NotImplementedError` stub; the JSON report already
  carries all the data a pivot table would need.
- `CallMode.json_guided` enum value: structured-output mode is not implemented and is
  now listed in LIMITATIONS rather than the schema docstring and project description.

[0.1.0]: https://github.com/SamLatino/callcheck/releases/tag/v0.1.0
