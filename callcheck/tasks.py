"""Task definitions and loaders for callcheck conformance tasks.

A 'task' is a single test case: a prompt, an expected schema or tool spec,
and metadata about what the model should produce.  Tasks are authored as YAML
files under tasks/ and loaded here into typed Pydantic models.

TODO:
  - TaskSpec pydantic model (id, prompt, tool_specs, expected_schema, tags)
  - load_tasks(path) -> list[TaskSpec]: scan a tasks/ directory and parse YAML
  - filter_tasks(tasks, tags) -> list[TaskSpec]: tag-based filtering
  - validate_task_yaml(raw): JSON Schema validation before model parse
"""
