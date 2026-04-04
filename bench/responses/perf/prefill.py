"""Prefill (prompt-processing) scenarios for POST /v1/responses.

Registers pp128/512/1024/2048/4096 plus batched variants pp*bN
(N in {1,2,4,8}). Each sends a filler prompt sized to hit the target token
count and requests minimal output. The key metric is prefill t/s.
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

_MAX_OUT = 3
_WORD = "one two three four five six seven eight nine ten "

# (target_tokens, n_full_reps, n_extra_words) — calibrated on Qwen3.5-0.8B.
_SIZES = [
    (128,  11, 4),
    (512,  49, 4),
    (1024, 100, 6),
    (2048, 203, 0),
    (4096, 407, 8),
]
_BATCH_SIZES = [1, 2, 4, 8]


def _make_input(n_reps: int, n_extra: int) -> str:
    base = _WORD * n_reps
    if n_extra:
        base += " ".join(_WORD.split()[:n_extra])
    return base


def _run_parallel_round(url: str, body: dict, concurrency: int) -> tuple[dict, float]:
    """Run one prefill benchmark round with *concurrency* in-flight requests."""
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
    max_ttft_ms = max((m.get("ttft_ms", 0) for m in req_metrics), default=0.0)
    model_info = req_metrics[0].get("model_info", {}) if req_metrics else {}

    # Aggregate using the max per-request phase time as the cohort window.
    max_prefill_ms = max((m.get("prefill_ms", 0) for m in req_metrics), default=0.0)
    max_decode_s = max((m.get("decode_s", 0) for m in req_metrics), default=0.0)
    prefill_s = max_prefill_ms / 1000.0 if max_prefill_ms > 0 else wall_s
    decode_s = max_decode_s if max_decode_s > 0 else wall_s

    agg = {
        "engine_tok_s": round(total_output_tokens / decode_s, 1) if decode_s > 0 else 0.0,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "decode_s": round(decode_s, 3),
        "prefill_tok_s": round(total_input_tokens / prefill_s, 1) if prefill_s > 0 else 0.0,
        "prefill_ms": round(max_prefill_ms, 1),
        "ttft_ms": round(max_ttft_ms, 1),
        "model_info": model_info,
    }
    return agg, wall_s


def _make_run(target: int, filler: str, concurrency: int):
    """Return a run() method closed over *target* and *filler*."""

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        url = f"{base_url}/v1/responses"
        model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
        precisions: list[str] = config.get("precision", ["original"])
        config["max_tokens"] = _MAX_OUT
        all_results: list[dict] = []

        for base_model in model_uris:
            for scheme in precisions:
                is_original = scheme == "original"
                uri = model_uri(base_model, None if is_original else scheme)
                print(f"\n  {uri}")

                body: dict = {
                    "model": uri,
                    "input": filler,
                    "instructions": "ok",
                    "stream": config.get("streaming", True),
                    "store": True,
                    "max_output_tokens": _MAX_OUT,
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
                    print(
                        f" prefill {m['prefill_tok_s']} tok/s | "
                        f"wall={round(wall_s, 3)}s input={m['input_tokens']}"
                        f" ttft={m.get('ttft_ms', 0):.1f}ms"
                        f" prefill_ms={m.get('prefill_ms', 0):.1f}"
                        f"{f' ({concurrency}x)' if concurrency > 1 else ''}"
                    )

        return all_results

    return run


# Dynamically register one Scenario subclass per (size, batch_size).
for _target, _reps, _extra in _SIZES:
    _filler = _make_input(_reps, _extra)
    for _batch_size in _BATCH_SIZES:
        _name = (
            f"responses/perf/pp{_target}"
            if _batch_size == 1
            else f"responses/perf/pp{_target}b{_batch_size}"
        )
        _cls_name = f"Pp{_target}" if _batch_size == 1 else f"Pp{_target}b{_batch_size}"
        _desc = (
            "Prefill throughput — minimal output."
            if _batch_size == 1
            else f"Prefill throughput — minimal output, {_batch_size} async in-flight."
        )
        _cls = type(_cls_name, (Scenario,), {
            "name": _name,
            "family": "pp",
            "description": _desc,
            "endpoint": "POST /v1/responses",
            "run": _make_run(_target, _filler, _batch_size),
        })

    # Alias: pp<size>b1 (same behavior as pp<size>) for explicit naming.
    _alias_name = f"responses/perf/pp{_target}b1"
    _alias_cls_name = f"Pp{_target}b1"
    _alias_cls = type(_alias_cls_name, (Scenario,), {
        "name": _alias_name,
        "family": "pp",
        "description": "Prefill throughput — minimal output, 1 async in-flight.",
        "endpoint": "POST /v1/responses",
        "run": _make_run(_target, _filler, 1),
    })
