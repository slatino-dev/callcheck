"""Tests for the tasks/ YAML corpus.

Verifies that every task YAML in the tasks/ directory:
  - loads without error
  - has a unique ID
  - passes basic structural invariants (non-empty messages, valid mode, etc.)
  - covers the expected categories via tags
"""

from __future__ import annotations

from pathlib import Path

import pytest

from callcheck.tasks import TaskSpec, filter_tasks, load_tasks

# Resolve the tasks directory relative to this test file's location
TASKS_DIR = Path(__file__).parent.parent / "tasks"


@pytest.fixture(scope="module")
def all_tasks() -> list[TaskSpec]:
    return load_tasks(TASKS_DIR)


class TestTaskCorpusLoading:
    def test_at_least_40_tasks(self, all_tasks: list[TaskSpec]) -> None:
        assert len(all_tasks) >= 40, f"Expected ≥40 tasks, got {len(all_tasks)}"

    def test_all_ids_unique(self, all_tasks: list[TaskSpec]) -> None:
        ids = [t.id for t in all_tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_all_tasks_have_at_least_one_message(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            assert len(task.messages) >= 1, f"Task {task.id} has no messages"

    def test_all_tasks_have_at_least_one_tag(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            assert len(task.tags) >= 1, f"Task {task.id} has no tags"

    def test_all_task_ids_are_non_empty_strings(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            assert isinstance(task.id, str) and task.id.strip(), "Task has empty ID"


class TestTaskCorpusCoverage:
    """Check that the corpus covers the required capability dimensions."""

    def _has_tag(self, tasks: list[TaskSpec], tag: str) -> list[TaskSpec]:
        return [t for t in tasks if tag in t.tags]

    def test_has_single_tool_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "single_tool")
        assert len(tagged) >= 5, f"Expected ≥5 single_tool tasks, got {len(tagged)}"

    def test_has_parallel_call_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "parallel")
        assert len(tagged) >= 3, f"Expected ≥3 parallel tasks, got {len(tagged)}"

    def test_has_no_call_needed_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "no_call_needed")
        assert len(tagged) >= 2, f"Expected ≥2 no_call_needed tasks, got {len(tagged)}"

    def test_has_schema_complex_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "schema_complex")
        assert len(tagged) >= 4, f"Expected ≥4 schema_complex tasks, got {len(tagged)}"

    def test_has_edge_case_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "edge_case")
        assert len(tagged) >= 5, f"Expected ≥5 edge_case tasks, got {len(tagged)}"

    def test_has_required_arg_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "required_arg")
        assert len(tagged) >= 5, f"Expected ≥5 required_arg tasks, got {len(tagged)}"

    def test_has_enum_arg_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "enum_arg")
        assert len(tagged) >= 2, f"Expected ≥2 enum_arg tasks, got {len(tagged)}"

    def test_has_unicode_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "unicode")
        assert len(tagged) >= 1, f"Expected ≥1 unicode tasks, got {len(tagged)}"

    def test_has_escaping_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "escaping")
        assert len(tagged) >= 2, f"Expected ≥2 escaping tasks, got {len(tagged)}"

    def test_has_nested_object_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "nested_object")
        assert len(tagged) >= 2, f"Expected ≥2 nested_object tasks, got {len(tagged)}"

    def test_has_numeric_arg_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "numeric_arg")
        assert len(tagged) >= 3, f"Expected ≥3 numeric_arg tasks, got {len(tagged)}"

    def test_has_array_arg_tasks(self, all_tasks: list[TaskSpec]) -> None:
        tagged = self._has_tag(all_tasks, "array_arg")
        assert len(tagged) >= 2, f"Expected ≥2 array_arg tasks, got {len(tagged)}"


class TestTaskCorpusStructure:
    """Per-task structural invariants."""

    def test_parallel_tasks_expect_count_gt_1(self, all_tasks: list[TaskSpec]) -> None:
        parallel = [t for t in all_tasks if "must_emit_n" in t.tags]
        for task in parallel:
            assert task.expect.tool_calls is not None
            assert task.expect.tool_calls.count > 1, (
                f"Task {task.id} tagged must_emit_n but count <= 1"
            )

    def test_no_call_tasks_expect_count_zero(self, all_tasks: list[TaskSpec]) -> None:
        nocall = [t for t in all_tasks if "no_call_needed" in t.tags]
        for task in nocall:
            assert task.expect.tool_calls is not None
            assert task.expect.tool_calls.count == 0, (
                f"Task {task.id} tagged no_call_needed but count != 0"
            )

    def test_tools_are_valid_dicts(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            for tool in task.tools:
                assert "type" in tool, f"Task {task.id}: tool missing 'type'"
                assert "function" in tool, f"Task {task.id}: tool missing 'function'"
                fn = tool["function"]
                assert "name" in fn, f"Task {task.id}: tool function missing 'name'"

    def test_tool_by_name_works_for_named_expectations(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            if task.expect.tool_calls and task.expect.tool_calls.name:
                expected_name = task.expect.tool_calls.name
                found = task.tool_by_name(expected_name)
                assert found is not None, (
                    f"Task {task.id} expects tool '{expected_name}' but it's not in tools list"
                )

    def test_filter_tasks_by_single_tool_tag(self, all_tasks: list[TaskSpec]) -> None:
        single_tasks = filter_tasks(all_tasks, ["single_tool"])
        assert all("single_tool" in t.tags for t in single_tasks)

    def test_filter_tasks_and_semantics(self, all_tasks: list[TaskSpec]) -> None:
        # Both single_tool AND required_arg
        filtered = filter_tasks(all_tasks, ["single_tool", "required_arg"])
        for task in filtered:
            assert "single_tool" in task.tags
            assert "required_arg" in task.tags

    def test_filter_by_no_call_needed(self, all_tasks: list[TaskSpec]) -> None:
        nocall = filter_tasks(all_tasks, ["no_call_needed"])
        assert len(nocall) >= 1
        for task in nocall:
            assert "no_call_needed" in task.tags


class TestTaskCorpusArgPredicates:
    """Verify that arg_predicates where set are syntactically valid."""

    def test_predicate_enum_values_are_lists(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            if task.expect.tool_calls:
                for key, pred in task.expect.tool_calls.arg_predicates.items():
                    if pred.enum is not None:
                        assert isinstance(pred.enum, list), (
                            f"Task {task.id}, arg '{key}': pred.enum must be a list"
                        )

    def test_predicate_min_max_are_numeric(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            if task.expect.tool_calls:
                for key, pred in task.expect.tool_calls.arg_predicates.items():
                    if pred.min is not None:
                        assert isinstance(pred.min, (int, float)), (
                            f"Task {task.id}, arg '{key}': pred.min must be numeric"
                        )
                    if pred.max is not None:
                        assert isinstance(pred.max, (int, float)), (
                            f"Task {task.id}, arg '{key}': pred.max must be numeric"
                        )

    def test_predicate_min_lte_max_when_both_set(self, all_tasks: list[TaskSpec]) -> None:
        for task in all_tasks:
            if task.expect.tool_calls:
                for key, pred in task.expect.tool_calls.arg_predicates.items():
                    if pred.min is not None and pred.max is not None:
                        assert pred.min <= pred.max, (
                            f"Task {task.id}, arg '{key}': pred.min > pred.max"
                        )
