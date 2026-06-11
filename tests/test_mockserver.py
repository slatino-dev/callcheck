"""Tests for the mockserver scoring pipeline.

These tests spin up the mockserver in-process (no real model needed) and verify
that callcheck's checker logic correctly classifies well-formed, malformed, and
refusal responses.

Currently: one passing smoke test that confirms the package imports and that
the mock response helpers produce valid OpenAI-schema dicts.
"""

from __future__ import annotations

import json


def _make_tool_call(name: str, arguments: str) -> dict:
    return {
        "id": "call_test001",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_completion(tool_calls: list | None = None, content: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


class TestMockResponseShape:
    """Verify that mock response helpers produce structurally valid completions."""

    def test_well_formed_tool_call_has_expected_keys(self) -> None:
        tc = _make_tool_call("get_weather", json.dumps({"location": "Dublin"}))
        completion = _make_completion(tool_calls=[tc])
        choice = completion["choices"][0]
        msg = choice["message"]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert args["location"] == "Dublin"

    def test_refusal_response_has_no_tool_calls(self) -> None:
        completion = _make_completion(content="I can't help with that.")
        msg = completion["choices"][0]["message"]
        assert msg["tool_calls"] is None
        assert msg["content"] == "I can't help with that."

    def test_malformed_arguments_are_not_valid_json(self) -> None:
        tc = _make_tool_call("get_weather", '{"location": "Dublin"')  # truncated
        completion = _make_completion(tool_calls=[tc])
        raw_args = completion["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        try:
            json.loads(raw_args)
            is_valid = True
        except json.JSONDecodeError:
            is_valid = False
        assert not is_valid, "Expected malformed arguments to fail JSON parse"

    def test_import_callcheck(self) -> None:
        import callcheck

        assert callcheck.__version__ == "0.1.0"
