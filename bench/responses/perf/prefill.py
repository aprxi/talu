"""Prefill (prompt-processing) scenarios for POST /v1/responses.

Registers pp512, pp1024, pp2048, pp4096.  Each sends a filler prompt sized
to hit the target token count (calibrated on Qwen3.5-0.8B; non-thinking
models will be ~4 tokens short) and requests minimal output.
The key metric is prefill t/s.
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

_MAX_OUT = 1
_WORD = "one two three four five six seven eight nine ten "

# (target_tokens, n_full_reps, n_extra_words) — calibrated on Qwen3.5-0.8B.
_SIZES = [
    (512,  49, 4),
    (1024, 100, 6),
    (2048, 203, 0),
    (4096, 407, 8),
]


def _make_input(n_reps: int, n_extra: int) -> str:
    base = _WORD * n_reps
    if n_extra:
        base += " ".join(_WORD.split()[:n_extra])
    return base


def _make_run(target: int, filler: str):
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
                    events, wall_s = http_post_stream(url, body)
                    m = extract_generation_metrics(events)
                    m["round"] = i + 1
                    m["scheme"] = scheme
                    m["model"] = base_model
                    m["model_uri"] = uri
                    m["wall_s"] = round(wall_s, 3)
                    all_results.append(m)
                    print(f" prefill {m['prefill_tok_s']} tok/s")

        return all_results

    return run


# Dynamically register one Scenario subclass per size.
for _target, _reps, _extra in _SIZES:
    _filler = _make_input(_reps, _extra)
    _cls = type(f"Pp{_target}", (Scenario,), {
        "name": f"responses/perf/pp{_target}",
        "family": "pp",
        "description": "Prefill throughput — minimal output.",
        "endpoint": "POST /v1/responses",
        "run": _make_run(_target, _filler),
    })
