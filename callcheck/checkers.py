"""Conformance checkers that score a model response against a TaskSpec.

Each checker is a pure function (or callable class) that takes a raw response
dict and a TaskSpec and returns a CheckResult with a pass/fail verdict plus
an optional detail message.  Checkers are composable — a task can require
multiple checks, all of which must pass.

TODO:
  - CheckResult dataclass (passed: bool, score: float, detail: str)
  - ToolCallChecker: verifies tool-call name + argument schema correctness
  - StructuredOutputChecker: validates response against a JSON Schema
  - RefusalChecker: detects unexpected refusals (no tool_calls + stop reason)
  - MalformedChecker: detects partial/broken JSON in tool-call arguments
  - run_checkers(response, task) -> list[CheckResult]
"""
