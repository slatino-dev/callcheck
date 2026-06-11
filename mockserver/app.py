"""OpenAI-compatible mock server for callcheck CI.

Exposes POST /v1/chat/completions.  The response behaviour is controlled by
a special system-message prefix in the request:

  "__mock__:ok"        -> well-formed single tool call (first tool in request)
  "__mock__:malformed" -> tool call with broken/truncated JSON arguments
  "__mock__:extra"     -> two tool calls when only one was expected
  "__mock__:refusal"   -> content reply, no tool_calls (simulates model refusal)
  "__mock__:empty_args"-> tool call present but arguments is empty string ""
  (default / anything else) -> well-formed single tool call

Usage:
  python -m mockserver.app          # runs on 127.0.0.1:9876
  python -m mockserver.app --port 9000

CI wires this up in conftest.py via a subprocess fixture so tests never need
a real vLLM endpoint.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mockserver requires 'fastapi' and 'uvicorn'.  "
        "Install them with: pip install fastapi uvicorn"
    ) from exc


app = FastAPI(title="callcheck-mockserver", version="0.1.0")


def _tool_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _completion(tool_calls: list[dict] | None = None, content: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        },
    }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _extract_mode(body: dict[str, Any]) -> str:
    """Return the __mock__ directive from the first system message, or 'ok'."""
    for msg in body.get("messages", []):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content.startswith("__mock__:"):
                return content.split(":", 1)[1].strip()
    return "ok"


def _first_tool_name(body: dict[str, Any]) -> str:
    tools = body.get("tools", [])
    if tools:
        return tools[0]["function"]["name"]
    return "unknown_tool"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body: dict[str, Any] = await request.json()
    mode = _extract_mode(body)
    tool_name = _first_tool_name(body)

    if mode == "malformed":
        # Deliberately broken JSON in arguments (truncated mid-string)
        tc = _tool_call(tool_name, '{"location": "Dublin"')  # missing closing }
        return JSONResponse(_completion(tool_calls=[tc]))

    if mode == "extra":
        # Two tool calls when one was expected
        tc1 = _tool_call(tool_name, json.dumps({"location": "Dublin"}))
        tc2 = _tool_call(tool_name, json.dumps({"location": "London"}))
        return JSONResponse(_completion(tool_calls=[tc1, tc2]))

    if mode == "refusal":
        # Content-only reply, no tool_calls
        return JSONResponse(_completion(content="I can't help with that."))

    if mode == "empty_args":
        tc = _tool_call(tool_name, "")
        return JSONResponse(_completion(tool_calls=[tc]))

    # Default / "ok": well-formed single tool call
    tc = _tool_call(tool_name, json.dumps({"location": "Dublin", "units": "celsius"}))
    return JSONResponse(_completion(tool_calls=[tc]))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
