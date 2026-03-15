"""Token-generation (decode) scenarios for POST /v1/responses.

Registers tg128, tg256, tg512, tg1024.  Each sends the same short prompt
(64 input tokens on Qwen3-0.6B) and varies max_output_tokens.
The key metric is generation t/s.
"""

from scenario import Scenario, http_post_stream, extract_generation_metrics, model_uri

_API_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}

_INPUT = (
    "Write a detailed story about a knight exploring a vast underground"
    " kingdom. Describe the caverns, the creatures, the ancient ruins,"
    " and a mysterious queen in vivid detail."
)
_INSTRUCTIONS = (
    "You are a novelist. Write in flowing prose."
    " Never summarize. Never use bullet points."
)

_SIZES = [128, 256, 512]


def _make_run(max_out: int):
    """Return a run() method closed over *max_out*."""

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        url = f"{base_url}/v1/responses"
        model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
        precisions: list[str] = config.get("precision", ["original"])
        config["max_tokens"] = max_out
        all_results: list[dict] = []

        for base_model in model_uris:
            for scheme in precisions:
                is_original = scheme == "original"
                uri = model_uri(base_model, None if is_original else scheme)
                print(f"\n  {uri}")

                body: dict = {
                    "model": uri,
                    "input": _INPUT,
                    "instructions": _INSTRUCTIONS,
                    "stream": config.get("streaming", True),
                    "store": True,
                    "max_output_tokens": max_out,
                }
                for cfg_key, api_key in _API_FIELDS.items():
                    if cfg_key in config:
                        body[api_key] = config[cfg_key]

                for i in range(rounds):
                    print(f"    round {i+1}/{rounds} ...", end="", flush=True)
                    events, wall_s = http_post_stream(url, body)
                    m = extract_generation_metrics(events)
                    m["round"] = i + 1
                    m["scheme"] = scheme
                    m["model"] = base_model
                    m["model_uri"] = uri
                    m["wall_s"] = round(wall_s, 3)
                    all_results.append(m)
                    print(f" {m['engine_tok_s']} tok/s")

        return all_results

    return run


# Dynamically register one Scenario subclass per size.
for _size in _SIZES:
    _cls = type(f"Tg{_size}", (Scenario,), {
        "name": f"responses/perf/tg{_size}",
        "family": "tg",
        "description": "Decode throughput — 64-token prompt.",
        "endpoint": "POST /v1/responses",
        "run": _make_run(_size),
    })
