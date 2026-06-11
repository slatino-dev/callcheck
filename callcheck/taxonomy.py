"""Failure taxonomy and tag registry for callcheck.

Provides a controlled vocabulary of failure modes so that reports can bucket
results into meaningful categories (e.g. "argument_type_mismatch",
"missing_required_arg", "extra_tool_call", "hallucinated_tool_name").

TODO:
  - FailureKind enum (exhaustive list of named failure modes)
  - TAGS registry: dict mapping string tag names to human descriptions
  - tag_result(check_results) -> list[FailureKind]: map CheckResults to kinds
  - severity(kind: FailureKind) -> Literal["critical", "major", "minor"]
"""
