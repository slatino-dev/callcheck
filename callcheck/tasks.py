"""Task definitions and loaders for callcheck conformance tasks.

A 'task' is a single test case: a prompt, an expected schema or tool spec,
and metadata about what the model should produce.  Tasks are authored as YAML
files under tasks/ and loaded here into typed Pydantic models.

Task YAML schema::

    id: str
    description: str (optional)
    messages: [{role: str, content: str}]
    tools: [OpenAI tool definition dict]
    mode: "native_toolcall" | "json_guided"   (default: native_toolcall)
    expect:
      tool_calls:
        count: int               # exact number expected (default: 1)
        name: str                # expected tool name
        required_args: [str]     # arg keys that must be present
        arg_predicates:          # optional key -> {type, enum, min, max}
          <key>: {type: str, enum: [...], min: ..., max: ...}
        arg_schema:              # optional per-arg JSON Schema fragments
          <key>: {type: str, ...}
    tags: [str]
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class CallMode(StrEnum):
    """How the model is instructed to produce tool calls."""

    native_toolcall = "native_toolcall"
    json_guided = "json_guided"


class ArgPredicate(BaseModel):
    """Optional per-argument predicate used in semantic checks."""

    type: str | None = None
    enum: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None


class ToolCallExpectation(BaseModel):
    """What the checker expects about the model's tool_calls."""

    count: int = 1
    name: str | None = None
    required_args: list[str] = Field(default_factory=list)
    # Per-arg predicates (richer semantic checks)
    arg_predicates: dict[str, ArgPredicate] = Field(default_factory=dict)
    # Per-arg JSON Schema fragments (passed directly to jsonschema)
    arg_schema: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Expectation(BaseModel):
    """Top-level expect block in a task YAML."""

    tool_calls: ToolCallExpectation | None = None


class TaskSpec(BaseModel):
    """A single callcheck conformance task."""

    id: str
    description: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    mode: CallMode = CallMode.native_toolcall
    expect: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one_message(self) -> TaskSpec:
        if not self.messages:
            raise ValueError(f"Task '{self.id}' must have at least one message")
        return self

    def tool_by_name(self, name: str) -> dict[str, Any] | None:
        """Return the tool definition dict for *name*, or None."""
        for t in self.tools:
            fn = t.get("function", {})
            if fn.get("name") == name:
                return t
        return None


def _validate_task_yaml(raw: dict[str, Any]) -> None:
    """Raise ValueError if *raw* is missing mandatory keys."""
    if "id" not in raw:
        raise ValueError("Task YAML must contain an 'id' field")
    if "messages" not in raw:
        raise ValueError(f"Task '{raw.get('id')}' must contain a 'messages' list")


def load_task(path: Path | str) -> TaskSpec:
    """Load a single task from a YAML file and return a :class:`TaskSpec`."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty or null YAML in {path}")
    _validate_task_yaml(raw)
    return TaskSpec.model_validate(raw)


def load_tasks(directory: Path | str) -> list[TaskSpec]:
    """Recursively scan *directory* for ``*.yaml`` / ``*.yml`` task files.

    Files are loaded in sorted order so the result is deterministic.
    Invalid files raise :class:`ValueError` immediately.
    """
    directory = Path(directory)
    tasks: list[TaskSpec] = []
    for yaml_path in sorted(directory.rglob("*.yaml")) + sorted(directory.rglob("*.yml")):
        tasks.append(load_task(yaml_path))
    return tasks


def filter_tasks(tasks: list[TaskSpec], tags: list[str] | None) -> list[TaskSpec]:
    """Return tasks that match ALL of *tags* (AND semantics).

    If *tags* is empty or None, all tasks are returned.
    """
    if not tags:
        return tasks
    tag_set = set(tags)
    return [t for t in tasks if tag_set.issubset(set(t.tags))]
