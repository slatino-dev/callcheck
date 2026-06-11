"""HTTP client for OpenAI-compatible model endpoints (vLLM, mockserver, etc.).

Wraps httpx with retry logic, timeout configuration, and response normalisation
so that the rest of callcheck never has to deal with raw HTTP concerns.

TODO:
  - ModelClient class: base_url, api_key, timeout, max_retries
  - complete(messages, tools, tool_choice, model) -> dict: raw API response
  - async acomplete(...): async variant using httpx.AsyncClient
  - Retry logic: exponential back-off on 429 / 5xx
  - Response normalisation: unify chat.completions schema across vLLM versions
"""
