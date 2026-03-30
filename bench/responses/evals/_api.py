"""API format translation for eval runner.

Translates a canonical eval request into either:
  - v1/responses (OpenResponses / talu serve)
  - v1/chat/completions (OpenAI compatible / vLLM)

Evals provide a canonical dict:
  {system, input, model, max_completion_tokens, max_reasoning_tokens, ...sampling}

The runner calls format_request() to get the API-specific body and endpoint path.
After the response, extract_output() returns the raw text regardless of API format.
"""

from __future__ import annotations


def format_request(canonical: dict, *, completions: bool = False) -> tuple[str, dict]:
    """Convert canonical eval request to API-specific (path, body).

    Args:
        canonical: Eval request with keys: model, system, input,
                   max_completion_tokens, max_reasoning_tokens, and
                   optional sampling params (temperature, top_p, etc).
        completions: If True, format for v1/chat/completions.
                     If False (default), format for v1/responses.

    Returns:
        (path, body) tuple ready for HTTP POST.
    """
    if completions:
        return _format_completions(canonical)
    return _format_responses(canonical)


def extract_output(response_data, *, completions: bool = False) -> dict:
    """Extract raw_output and reasoning from API response.

    Args:
        response_data: Parsed response (events list for responses,
                       dict for completions).
        completions: Which API format to parse.

    Returns:
        {"raw_output": str, "reasoning": str, "response_output": list,
         "input_tokens": int, "output_tokens": int, ...metrics}
    """
    if completions:
        # Completions response is plain JSON. The runner's _parse_sse wraps
        # it as {"event": "response.completed", "data": {"response": json}}.
        # Unwrap to get the actual completions response.
        if isinstance(response_data, list):
            for ev in response_data:
                data = ev.get("data", {})
                # _parse_sse wraps in {"response": ...}
                resp = data.get("response", data)
                if resp:
                    return _extract_completions(resp)
            return _extract_completions({})
        return _extract_completions(response_data)
    return _extract_responses(response_data)


# ---------------------------------------------------------------------------
# v1/responses (OpenResponses)
# ---------------------------------------------------------------------------

_RESPONSES_PASSTHROUGH = {
    "temperature", "top_p", "top_k", "seed", "presence_penalty",
    "frequency_penalty",
}


def _format_responses(c: dict) -> tuple[str, dict]:
    body: dict = {
        "model": c["model"],
        "input": c["input"],
        "stream": False,
        "store": c.get("store", False),
    }
    if c.get("system"):
        body["instructions"] = c["system"]
    if c.get("max_completion_tokens") is not None:
        body["max_completion_tokens"] = c["max_completion_tokens"]
    if c.get("max_reasoning_tokens") is not None:
        body["max_reasoning_tokens"] = c["max_reasoning_tokens"]
    if c.get("max_output_tokens") is not None:
        body["max_output_tokens"] = c["max_output_tokens"]
    if c.get("tools"):
        body["tools"] = c["tools"]
    for key in _RESPONSES_PASSTHROUGH:
        if key in c:
            body[key] = c[key]
    return "/v1/responses", body


def _extract_responses(events: list[dict]) -> dict:
    raw_output = ""
    reasoning = ""
    response_output: list = []
    input_tokens = 0
    output_tokens = 0

    for ev in events:
        if ev.get("event") in ("response.completed", "response.incomplete"):
            resp = ev.get("data", {}).get("response", {})
            response_output = resp.get("output", [])
            usage = resp.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            for item in response_output:
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") in ("output_text", "text"):
                            raw_output += part.get("text", "")
                elif item.get("type") == "reasoning":
                    for part in item.get("content", []):
                        if part.get("type") in ("reasoning_text", "text"):
                            reasoning += part.get("text", "")
                    if not reasoning:
                        for s in item.get("summary", []):
                            if s.get("type") == "summary_text":
                                reasoning += s.get("text", "")

    return {
        "raw_output": raw_output.strip(),
        "reasoning": reasoning,
        "response_output": response_output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# v1/chat/completions (OpenAI compatible)
# ---------------------------------------------------------------------------

_COMPLETIONS_PASSTHROUGH = {
    "temperature", "top_p", "top_k", "seed", "presence_penalty",
    "frequency_penalty",
}


def _wrap_tool(t: dict) -> dict:
    """Convert flat tool format to OpenAI completions nested format."""
    if "function" in t:
        return t  # Already in nested format
    func = {k: v for k, v in t.items() if k != "type"}
    return {"type": "function", "function": func}


def _format_completions(c: dict) -> tuple[str, dict]:
    messages = []
    if c.get("system"):
        messages.append({"role": "system", "content": c["system"]})
    messages.append({"role": "user", "content": c["input"]})

    body: dict = {
        "model": c["model"],
        "messages": messages,
    }
    # Map max_completion_tokens → max_tokens for completions API.
    max_tokens = c.get("max_completion_tokens") or c.get("max_output_tokens")
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if c.get("tools"):
        # OpenAI completions format: {"type":"function","function":{"name":...,"parameters":...}}
        # Canonical (responses) format: {"type":"function","name":...,"parameters":...}
        body["tools"] = [_wrap_tool(t) for t in c["tools"]]
    if c.get("tool_choice"):
        body["tool_choice"] = c["tool_choice"]
    for key in _COMPLETIONS_PASSTHROUGH:
        if key in c:
            body[key] = c[key]
    return "/v1/chat/completions", body


def _extract_completions(resp: dict) -> dict:
    raw_output = ""
    reasoning = ""
    input_tokens = 0
    output_tokens = 0
    response_output: list = []

    usage = resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    for choice in resp.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if content:
            raw_output = content.strip()
        # Some APIs put reasoning in reasoning_content field.
        rc = msg.get("reasoning_content", "")
        if rc:
            reasoning = rc
        # Extract tool_calls and convert to responses-format function_call items
        # so BFCL's _extract_tool_calls can read them uniformly.
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")
                response_output.append({
                    "type": "function_call",
                    "name": name,
                    "arguments": arguments,
                })

    return {
        "raw_output": raw_output,
        "reasoning": reasoning,
        "response_output": response_output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
