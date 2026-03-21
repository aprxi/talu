"""Token-generation (decode) scenarios for POST /v1/responses.

Registers tg128, tg256, tg512, tg1024.  Each sends the same short prompt
(64 input tokens on Qwen3-0.6B) and varies max_output_tokens.
The key metric is generation t/s.
"""

import concurrent.futures
import time

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
_BATCH_SIZES = [1, 2, 4, 8]


def _run_parallel_round(url: str, body: dict, concurrency: int) -> tuple[dict, float]:
    """Run one benchmark round with *concurrency* in-flight requests.

    Returns (aggregated_metrics, wall_seconds) where aggregated metrics represent
    total throughput across all requests in the round.
    """
    t0 = time.monotonic()
    req_metrics: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(http_post_stream, url, body) for _ in range(concurrency)]
        for fut in concurrent.futures.as_completed(futures):
            events, _ = fut.result()
            req_metrics.append(extract_generation_metrics(events))

    wall_s = time.monotonic() - t0
    total_input_tokens = sum(m.get("input_tokens", 0) for m in req_metrics)
    total_output_tokens = sum(m.get("output_tokens", 0) for m in req_metrics)
    avg_ttft_ms = (
        sum(m.get("ttft_ms", 0) for m in req_metrics) / len(req_metrics)
        if req_metrics else 0.0
    )
    model_info = req_metrics[0].get("model_info", {}) if req_metrics else {}

    agg = {
        "engine_tok_s": round(total_output_tokens / wall_s, 1) if wall_s > 0 else 0.0,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "decode_s": round(wall_s, 3),
        "prefill_tok_s": round(total_input_tokens / wall_s, 1) if wall_s > 0 else 0.0,
        "prefill_ms": 0.0,
        "ttft_ms": round(avg_ttft_ms, 1),
        "model_info": model_info,
    }
    return agg, wall_s


def _make_run(max_out: int, concurrency: int):
    """Return a run() method closed over *max_out* and *concurrency*."""

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
                    if concurrency == 1:
                        events, wall_s = http_post_stream(url, body)
                        m = extract_generation_metrics(events)
                    else:
                        m, wall_s = _run_parallel_round(url, body, concurrency)
                    m["round"] = i + 1
                    m["scheme"] = scheme
                    m["model"] = base_model
                    m["model_uri"] = uri
                    m["batch_size"] = concurrency
                    m["wall_s"] = round(wall_s, 3)
                    all_results.append(m)
                    if concurrency == 1:
                        print(f" {m['engine_tok_s']} tok/s | wall={round(wall_s, 3)}s decode={m['decode_s']}s tokens={m['output_tokens']}")
                    else:
                        print(f" {m['engine_tok_s']} tok/s | wall={round(wall_s, 3)}s decode={m['decode_s']}s tokens={m['output_tokens']} ({concurrency}x)")

        return all_results

    return run


# Dynamically register one Scenario subclass per (size, batch_size).
for _size in _SIZES:
    for _batch_size in _BATCH_SIZES:
        _name = (
            f"responses/perf/tg{_size}"
            if _batch_size == 1
            else f"responses/perf/tg{_size}b{_batch_size}"
        )
        _cls_name = f"Tg{_size}" if _batch_size == 1 else f"Tg{_size}b{_batch_size}"
        _desc = (
            "Decode throughput — 64-token prompt."
            if _batch_size == 1
            else f"Decode throughput — 64-token prompt, {_batch_size} async in-flight."
        )
        _cls = type(_cls_name, (Scenario,), {
            "name": _name,
            "family": "tg",
            "description": _desc,
            "endpoint": "POST /v1/responses",
            "run": _make_run(_size, _batch_size),
        })

    # Alias: tg<size>b1 (same behavior as tg<size>) for explicit naming.
    _alias_name = f"responses/perf/tg{_size}b1"
    _alias_cls_name = f"Tg{_size}b1"
    _alias_cls = type(_alias_cls_name, (Scenario,), {
        "name": _alias_name,
        "family": "tg",
        "description": "Decode throughput — 64-token prompt, 1 async in-flight.",
        "endpoint": "POST /v1/responses",
        "run": _make_run(_size, 1),
    })
