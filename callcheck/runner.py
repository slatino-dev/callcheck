"""Test runner: drives tasks through a ModelClient and collects scored results.

This module owns the evaluation loop for a *single matrix cell* — one
(model, base_url, parser_backend) combination run against a task suite.
Orchestration of the full benchmark matrix (e.g. restarting vLLM between
models) happens in :mod:`scripts.spark_matrix` on the bench host; this
module is deliberately ignorant of that layer.

Key features
------------
* **k-repeat agreement** — each task is sent ``k`` times; per-task
  :attr:`RunResult.agreement` reports what fraction of runs agreed on
  pass/fail, surface the instability of borderline cases.
* **Resumable checkpointing** — if ``checkpoint_path`` is supplied the
  runner saves partial results as a JSONL file and skips already-evaluated
  tasks on restart (useful when a run is interrupted mid-suite).
* **Progress callback** — callers (e.g. a Rich live display) register a
  ``ProgressCallback`` that fires after every task completion.
* **Async variant** — :func:`async_run_suite` parallelises task dispatch up
  to ``concurrency`` in-flight tasks using :mod:`asyncio`.

Typical usage::

    from callcheck.runner import RunConfig, run_suite
    from callcheck.tasks import load_tasks

    tasks = load_tasks("tasks/")
    config = RunConfig(model="qwen3-8b", base_url="http://127.0.0.1:9876")
    results = run_suite(tasks, config)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from callcheck.checkers import CheckResult, run_checkers
from callcheck.client import AsyncModelClient, ModelClient
from callcheck.tasks import TaskSpec
from callcheck.taxonomy import FailureKind, classify_results, primary_kind

# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

#: Signature: ``callback(task_id, result, done, total)``
ProgressCallback = Callable[["RunResult", int, int], None]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for a single matrix-cell evaluation run.

    Parameters
    ----------
    model:
        Model identifier sent to the endpoint (e.g. ``"qwen3-8b"``).
    base_url:
        Root URL of the OpenAI-compatible API server.  Defaults to
        ``OPENAI_BASE_URL`` / ``LLM_BASE_URL`` env vars, then
        ``http://127.0.0.1:9876``.
    api_key:
        Bearer token.  Defaults to ``OPENAI_API_KEY`` env var, then
        ``"none"``.
    parser_backend:
        Metadata label for the tool-call parser in use (e.g.
        ``"hermes-2"``, ``"mistral"``).  Not sent to the model; recorded
        in results for matrix comparison.
    k:
        Number of repeat calls per task.  Agreement across repeats is
        tracked in :attr:`RunResult.agreement`.
    concurrency:
        Max in-flight async tasks (used by :func:`async_run_suite`).
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retries on 429 / 5xx responses.
    checkpoint_path:
        If provided, partial results are appended to this JSONL file and
        already-evaluated task IDs are skipped on restart.
    tags:
        Only run tasks whose tags are a superset of this list (AND
        semantics).  Empty list means run all tasks.
    temperature:
        Sampling temperature.  Keep at 0.0 for deterministic eval; raise
        slightly (e.g. 0.2) when testing agreement variance.
    """

    model: str = "default"
    base_url: str | None = None
    api_key: str | None = None
    parser_backend: str = "default"
    k: int = 1
    concurrency: int = 4
    timeout: float = 60.0
    max_retries: int = 2
    checkpoint_path: Path | None = None
    tags: list[str] = field(default_factory=list)
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregated result of running one task (k repeats).

    Attributes
    ----------
    task_id:
        The :attr:`~callcheck.tasks.TaskSpec.id` of the evaluated task.
    passed:
        ``True`` if **all** k runs passed every checker.
    agreement:
        Fraction of k runs that produced the same pass/fail verdict as the
        majority (1.0 = perfect agreement; <1.0 = instability).
    check_results:
        Checker results from the *last* run (useful for failure detail).
    failure_kinds:
        Ordered list of :class:`~callcheck.taxonomy.FailureKind` labels
        from the last run (empty when passed).
    primary_failure:
        The highest-priority :class:`~callcheck.taxonomy.FailureKind`, or
        ``None`` on pass.
    raw_responses:
        All k raw API response dicts (last-in list corresponds to
        ``check_results``).
    latencies_ms:
        Round-trip latency in milliseconds for each of the k calls.
    tokens_prompt:
        Prompt token counts across k runs (from usage field).
    tokens_completion:
        Completion token counts across k runs.
    model:
        Model identifier (from :class:`RunConfig`).
    parser_backend:
        Parser backend label (from :class:`RunConfig`).
    task_tags:
        Tags from the :class:`~callcheck.tasks.TaskSpec`.
    """

    task_id: str
    passed: bool
    agreement: float
    check_results: list[CheckResult]
    failure_kinds: list[FailureKind]
    primary_failure: FailureKind | None
    raw_responses: list[dict[str, Any]]
    latencies_ms: list[float]
    tokens_prompt: list[int]
    tokens_completion: list[int]
    model: str
    parser_backend: str
    task_tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience aggregates
    # ------------------------------------------------------------------

    @property
    def mean_latency_ms(self) -> float:
        """Mean round-trip latency in ms across all k runs."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def mean_tokens_prompt(self) -> float:
        if not self.tokens_prompt:
            return 0.0
        return sum(self.tokens_prompt) / len(self.tokens_prompt)

    @property
    def mean_tokens_completion(self) -> float:
        if not self.tokens_completion:
            return 0.0
        return sum(self.tokens_completion) / len(self.tokens_completion)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (for checkpointing and export)."""
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "agreement": self.agreement,
            "failure_kinds": [k.value for k in self.failure_kinds],
            "primary_failure": self.primary_failure.value if self.primary_failure else None,
            "latencies_ms": self.latencies_ms,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "model": self.model,
            "parser_backend": self.parser_backend,
            "task_tags": self.task_tags,
            # Omit raw_responses and check_results from checkpoint — they
            # can be large and are only needed for live drill-down.
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> set[str]:
    """Return the set of task_ids already recorded in *path*."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                seen.add(row["task_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def _append_checkpoint(path: Path, result: RunResult) -> None:
    """Append *result* as a JSON line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Single-task evaluation (synchronous)
# ---------------------------------------------------------------------------


def _run_task_once(
    task: TaskSpec,
    client: ModelClient,
    config: RunConfig,
) -> tuple[bool, list[CheckResult], dict[str, Any], float, int, int]:
    """Send *task* once to the model and score it.

    Returns ``(passed, check_results, raw_response, latency_ms, tok_prompt, tok_completion)``.
    """
    t0 = time.monotonic()
    response = client.complete(
        messages=task.messages,
        tools=task.tools if task.tools else None,
        model=config.model,
        temperature=config.temperature,
    )
    latency_ms = (time.monotonic() - t0) * 1000.0

    results = run_checkers(response, task)
    passed = all(r.passed for r in results)

    usage = response.get("usage") or {}
    tok_prompt = int(usage.get("prompt_tokens") or 0)
    tok_completion = int(usage.get("completion_tokens") or 0)

    return passed, results, response, latency_ms, tok_prompt, tok_completion


def run_task(task: TaskSpec, client: ModelClient, config: RunConfig) -> RunResult:
    """Evaluate *task* with k repeats and return an aggregated :class:`RunResult`.

    Parameters
    ----------
    task:
        The task to evaluate.
    client:
        A configured :class:`~callcheck.client.ModelClient`.
    config:
        Run configuration (model, k, temperature, …).
    """
    pass_count = 0
    all_check_results: list[CheckResult] = []
    raw_responses: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    tokens_prompt: list[int] = []
    tokens_completion: list[int] = []

    for _ in range(config.k):
        passed, checks, response, lat, tp, tc = _run_task_once(task, client, config)
        if passed:
            pass_count += 1
        all_check_results = checks  # keep last run for detail
        raw_responses.append(response)
        latencies_ms.append(lat)
        tokens_prompt.append(tp)
        tokens_completion.append(tc)

    # Agreement: fraction of runs matching the majority verdict
    majority_passes = pass_count >= (config.k / 2)
    agree_count = pass_count if majority_passes else (config.k - pass_count)
    agreement = agree_count / config.k if config.k > 0 else 1.0

    final_passed = pass_count == config.k  # strict: all k must pass
    failure_kinds = classify_results(all_check_results)

    return RunResult(
        task_id=task.id,
        passed=final_passed,
        agreement=agreement,
        check_results=all_check_results,
        failure_kinds=failure_kinds,
        primary_failure=primary_kind(failure_kinds),
        raw_responses=raw_responses,
        latencies_ms=latencies_ms,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model=config.model,
        parser_backend=config.parser_backend,
        task_tags=list(task.tags),
    )


# ---------------------------------------------------------------------------
# Suite runner (synchronous, resumable)
# ---------------------------------------------------------------------------


def run_suite(
    tasks: list[TaskSpec],
    config: RunConfig,
    progress: ProgressCallback | None = None,
) -> list[RunResult]:
    """Run the full task suite and return all :class:`RunResult` objects.

    The runner is resumable: if ``config.checkpoint_path`` is set, tasks
    whose IDs already appear in the checkpoint file are skipped and a
    placeholder :class:`RunResult` is **not** injected (callers reading the
    checkpoint should load those themselves via :func:`load_checkpoint_results`).

    Parameters
    ----------
    tasks:
        Task list, typically from :func:`~callcheck.tasks.load_tasks`.
    config:
        Run configuration.
    progress:
        Optional callback fired after each task: ``progress(result, done, total)``.
    """
    client = ModelClient(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    seen: set[str] = set()
    if config.checkpoint_path is not None:
        seen = _load_checkpoint(config.checkpoint_path)

    pending = [t for t in tasks if t.id not in seen] if seen else tasks

    results: list[RunResult] = []
    total = len(pending)

    for idx, task in enumerate(pending, start=1):
        result = run_task(task, client, config)
        results.append(result)

        if config.checkpoint_path is not None:
            _append_checkpoint(config.checkpoint_path, result)

        if progress is not None:
            progress(result, idx, total)

    return results


# ---------------------------------------------------------------------------
# Async suite runner (parallel)
# ---------------------------------------------------------------------------


async def _run_task_async(
    task: TaskSpec,
    client: AsyncModelClient,
    config: RunConfig,
) -> RunResult:
    """Async variant of :func:`run_task` (k repeats via AsyncModelClient)."""
    pass_count = 0
    all_check_results: list[CheckResult] = []
    raw_responses: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    tokens_prompt: list[int] = []
    tokens_completion: list[int] = []

    for _ in range(config.k):
        t0 = time.monotonic()
        response = await client.acomplete(
            messages=task.messages,
            tools=task.tools if task.tools else None,
            model=config.model,
            temperature=config.temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        results = run_checkers(response, task)
        passed = all(r.passed for r in results)

        if passed:
            pass_count += 1
        all_check_results = results
        raw_responses.append(response)
        latencies_ms.append(latency_ms)
        usage = response.get("usage") or {}
        tokens_prompt.append(int(usage.get("prompt_tokens") or 0))
        tokens_completion.append(int(usage.get("completion_tokens") or 0))

    majority_passes = pass_count >= (config.k / 2)
    agree_count = pass_count if majority_passes else (config.k - pass_count)
    agreement = agree_count / config.k if config.k > 0 else 1.0
    final_passed = pass_count == config.k
    failure_kinds = classify_results(all_check_results)

    return RunResult(
        task_id=task.id,
        passed=final_passed,
        agreement=agreement,
        check_results=all_check_results,
        failure_kinds=failure_kinds,
        primary_failure=primary_kind(failure_kinds),
        raw_responses=raw_responses,
        latencies_ms=latencies_ms,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model=config.model,
        parser_backend=config.parser_backend,
        task_tags=list(task.tags),
    )


async def async_run_suite(
    tasks: list[TaskSpec],
    config: RunConfig,
    progress: ProgressCallback | None = None,
) -> list[RunResult]:
    """Run tasks concurrently (up to ``config.concurrency`` in-flight).

    Results are returned in the same order as *tasks*.

    Parameters
    ----------
    tasks:
        Task list.
    config:
        Run configuration (``concurrency`` controls parallelism).
    progress:
        Optional callback fired after each task completes.
    """
    client = AsyncModelClient(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    seen: set[str] = set()
    if config.checkpoint_path is not None:
        seen = _load_checkpoint(config.checkpoint_path)

    pending_indices = [i for i, t in enumerate(tasks) if t.id not in seen]
    pending_tasks = [tasks[i] for i in pending_indices]
    total = len(pending_tasks)

    semaphore = asyncio.Semaphore(config.concurrency)
    done_count = 0
    results_by_index: dict[int, RunResult] = {}

    async def _bounded(idx: int, task: TaskSpec) -> tuple[int, RunResult]:
        async with semaphore:
            return idx, await _run_task_async(task, client, config)

    coros = [_bounded(orig_idx, task) for orig_idx, task in zip(pending_indices, pending_tasks, strict=True)]

    for coro in asyncio.as_completed(coros):
        orig_idx, result = await coro
        results_by_index[orig_idx] = result
        done_count += 1

        if config.checkpoint_path is not None:
            _append_checkpoint(config.checkpoint_path, result)

        if progress is not None:
            progress(result, done_count, total)

    # Return in original task order
    return [results_by_index[i] for i in pending_indices]


# ---------------------------------------------------------------------------
# Checkpoint result loader
# ---------------------------------------------------------------------------


def load_checkpoint_results(path: Path) -> list[dict[str, Any]]:
    """Load all results previously written to a checkpoint JSONL file.

    Returns a list of dicts (the serialised form); callers may cast them
    to :class:`RunResult` objects if needed, but the dict form is sufficient
    for report generation.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ---------------------------------------------------------------------------
# spark_matrix placeholder
# ---------------------------------------------------------------------------


def spark_matrix_guide() -> str:
    """Return a human-readable description of how to run the full benchmark matrix.

    The actual matrix orchestration runs **on the bench host** because it
    requires restarting vLLM between models.  This function documents the
    intended loop so the bench script can import and print it.

    Bench-host restart loop (pseudocode)
    ------------------------------------
    For each (model, parser_backend) combination:

    1.  SSH to the bench host and restart vLLM with the new model::

            ssh bench-host \\
              "systemctl stop vllm && \\
               export MODEL_ID=<model> && \\
               systemctl start vllm && \\
               sleep 30 && \\
               curl http://localhost:8000/health"

    2.  Run callcheck against the live endpoint::

            callcheck run \\
              --model <model> \\
              --base-url http://bench-host:8000 \\
              --parser-backend <parser_backend> \\
              --tasks tasks/ \\
              --k 3 \\
              --checkpoint results/<model>_<parser>.jsonl \\
              --output results/<model>_<parser>_report.json

    3.  After all cells complete, aggregate with::

            callcheck report --matrix results/*.jsonl

    Matrix dimensions (illustrative — update when the bench roster changes)
    -----------------------------------------------------------------------
    models:
      - qwen2.5-7b-instruct
      - qwen2.5-72b-instruct
      - hermes-3-llama-3.1-8b
      - mistral-nemo-12b
      - llama-3.1-8b-instruct

    parser_backends:
      - hermes-2       (vLLM --tool-call-parser hermes)
      - mistral        (vLLM --tool-call-parser mistral)
      - llama3_json    (vLLM --tool-call-parser llama3_json)

    DO NOT fabricate model results.  The bench host runs real inference;
    this callcheck repo scores the outputs.
    """
    return spark_matrix_guide.__doc__ or ""
