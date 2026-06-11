# callcheck

Scores tool-calling conformance of OpenAI-compatible endpoints against a YAML task corpus.

## The problem

vLLM's tool-call-parser backends (`hermes`, `mistral`, `llama3_json`, etc.) each implement a
different heuristic for extracting structured JSON from model output.  A model that correctly
generates a tool call in one parser configuration silently corrupts, truncates, or hallucinates
parameters in another.  There is no standard score — every team re-derives the same ad-hoc
checklist when something breaks in production.

callcheck gives that checklist a name, a type, and a test suite.

## Architecture

```
tasks/*.yaml          YAML corpus (42 tasks, 5 capability categories)
     │
     ▼
ModelClient           httpx wrapper → POST /v1/chat/completions
     │                (sync + async variants; retry + Retry-After backoff)
     ▼
run_checkers()        10-stage pipeline per response
  1  check_no_call         — did the model call any tool?
  2  check_call_count      — is the call count right?
  3  check_tool_name       — right function name?
  4  check_json_parse      — do arguments parse as JSON?
  5  check_truncation      — are arguments empty / None?
  6  check_schema          — does the JSON satisfy the tool schema?
  7  check_required_args   — required keys present?
  8  check_hallucinated_params — unknown keys?
  9  check_arg_predicates  — semantic predicates (enum, range, pattern)?
  10 check_escaping        — double-escaped Unicode or control chars?
     │
     ▼
classify_results()    12-label failure taxonomy → primary FailureKind
     │
     ▼
RunResult             k-repeat agreement, latency, token counts, JSONL checkpoint
     │
     ▼
render()              Rich console table  OR  JSON export for CI artefacts
```

**Stack:** Python 3.11+, Pydantic v2, httpx, jsonschema, Rich, Typer, PyYAML.
CI: ruff (strict), mypy, pytest + pytest-asyncio, a secret-scrub gate.

## Task corpus

42 YAML tasks across five categories:

| Category | Count | What it tests |
|---|---|---|
| `single/` | varies | Basic single-call conformance, type coercion, enum values |
| `schema/` | varies | `$ref`, `anyOf`, float precision, embedded JSON strings, RTL text |
| `parallel/` | varies | Multi-call dispatch and parallel-collapse detection |
| `nocall/` | 2 | Refusal / conversational responses (model should NOT call any tool) |
| `edge/` | varies | Emoji in args, very long strings, forced `tool_choice` |

Corpus invariant: every `id` is unique; each category has a minimum task count (enforced by
`tests/test_task_corpus.py`).

## Failure taxonomy

12 controlled failure kinds with severity tiers and a deterministic priority ordering:

| Kind | Severity | Description |
|---|---|---|
| `no_call` | critical | Content-only response when a tool call was expected |
| `wrong_tool` | critical | Called a different function than requested |
| `malformed_json` | critical | `arguments` is not valid JSON |
| `truncation` | major | `arguments` is empty or None |
| `missing_required` | major | Required argument key absent |
| `schema_violation` | major | Arguments fail the tool's JSON Schema |
| `parallel_collapse` | major | Wrong number of calls (multi-call task collapsed or vice-versa) |
| `hallucinated_param` | major | Arguments contain a key not in the schema |
| `type_coercion` | minor | Value has wrong JSON type but is otherwise present |
| `escaping_error` | minor | Double-escaped Unicode or control characters in string values |
| `spurious_call` | minor | More calls than expected |
| `predicate_fail` | minor | Semantic predicate (enum, range, pattern) not satisfied |

When multiple kinds are present in one response the highest-priority label wins.

## Quickstart

```bash
pip install -e ".[dev]"

# Run against the bundled deterministic mockserver (no real model needed)
python -m mockserver.app &          # listens on 127.0.0.1:9876
callcheck run --tasks tasks/ --base-url http://127.0.0.1:9876 --model mock-model

# Run against a real vLLM endpoint
callcheck run \
  --tasks tasks/ \
  --base-url http://your-vllm-host:8000 \
  --model qwen2.5-7b-instruct \
  --parser-backend hermes \
  --k 3 \
  --checkpoint results/qwen25-7b_hermes.jsonl \
  --output results/qwen25-7b_hermes_report.json

# JSON report
callcheck run --tasks tasks/ --format json --output report.json
```

### Full benchmark matrix (bench-host loop)

Each `(model, parser_backend)` cell requires restarting vLLM; the loop runs on the bench host:

```bash
for MODEL in qwen2.5-7b-instruct qwen2.5-72b-instruct hermes-3-llama-3.1-8b mistral-nemo-12b; do
  for PARSER in hermes mistral llama3_json; do
    ssh bench-host \
      "systemctl stop vllm && MODEL_ID=${MODEL} PARSER=${PARSER} systemctl start vllm && sleep 30"
    callcheck run \
      --tasks tasks/ \
      --base-url http://bench-host:8000 \
      --model "${MODEL}" --parser-backend "${PARSER}" \
      --k 3 --checkpoint "results/${MODEL}_${PARSER}.jsonl" \
      --output "results/${MODEL}_${PARSER}_report.json"
  done
done
```

## Results

**Matrix results are pending real DGX runs.**  The mockserver-proved scorer is the shipped value
now: 168 deterministic tests exercise every checker and every taxonomy label, including the full
pipeline from task YAML to classified `RunResult`, verified in ~4 seconds on any machine.

The scorer does not make up numbers.

## LIMITATIONS

- **First-call-only validation.** Checkers 3–10 operate on the first tool call in the response.
  Parallel tasks (count ≥ 2) confirm the call count and name but do not schema-check calls 2+.
- **`json_guided` mode unimplemented.** `CallMode` accepts only `native_toolcall` today.
  Structured-output scoring (send `response_format`/`json_schema`, score `message.content` as
  JSON against a schema) is planned but absent.  It is listed in LIMITATIONS rather than the
  feature list.
- **`RunConfig.tags` filtering is CLI-only.** The `tags` field on `RunConfig` is not wired
  into `run_suite` / `async_run_suite`; tag filtering happens at the CLI layer before the suite
  is called.
- **k-agreement at k=1 is always 1.0.** The default `k=1` gives no instability signal.
  Raise `k` to 3–5 to surface borderline-case variance.
- **No streaming evaluation.** The client collects the full response before scoring.  Streaming
  tool-call assembly (where partial JSON appears in chunks) is out of scope.

## What I would do differently

The taxonomy's `parallel_collapse` label covers two distinct directions (N→1 and 1→N collapse)
that are conflated in the current `check_call_count` logic.  Splitting them into separate labels
would make failure dashboards more actionable.

`AsyncModelClient` now holds a persistent `httpx.AsyncClient` across requests; the suite runner
could expose it as an async context manager for explicit lifecycle control in longer benchmark
runs.  The API supports `async with AsyncModelClient(...) as client:` already but the runner
does not surface that to callers.
