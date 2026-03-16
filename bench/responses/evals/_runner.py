"""Shared eval runner loop with retry, resume, and throughput tracking."""

from __future__ import annotations

import time
from collections import deque

from scenario import http_post_stream, extract_generation_metrics, model_uri
from extract import extract_answer
from log import eval_log_path, load_completed, EvalLogger

_MAX_RETRIES = 3


def _extract_text(events: list[dict]) -> str:
    """Extract generated text from response events."""
    for ev in events:
        if ev["event"] in ("response.completed", "response.incomplete"):
            resp = ev["data"].get("response", {})
            for item in resp.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") in ("output_text", "text"):
                        return part.get("text", "").strip()
    return ""


def run_eval(
    *,
    bench_name: str,
    base_url: str,
    config: dict,
    samples: list[dict],
    build_body,
) -> list[dict]:
    """Run an evaluation loop over samples for each model × precision.

    Args:
        bench_name: Benchmark identifier (e.g. "mmlu").
        base_url: Server base URL.
        config: Loaded config dict.
        samples: List of sample dicts with at least: prompt, correct, question_hash, index.
        build_body: Callable(sample, uri, config) -> request body dict.
                    For multimodal, this handles image upload etc.

    Returns:
        List of result dicts (one per model × precision).
    """
    url = f"{base_url}/v1/responses"
    model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
    precisions: list[str] = config.get("precision", ["original"])
    samples_n: int | None = config.get("samples")
    if isinstance(samples_n, str):
        samples_n = int(samples_n)
    total = len(samples)

    all_results: list[dict] = []

    for base_model in model_uris:
        for scheme in precisions:
            is_original = scheme == "original"
            uri = model_uri(base_model, None if is_original else scheme)

            # Resume support.
            log_path = eval_log_path(bench_name, uri, samples_n)
            completed, cached_stats = load_completed(log_path)
            logger = EvalLogger(log_path)
            cached = sum(1 for (m, _) in completed if m == uri)
            cached_correct = cached_stats["cached_correct"]
            if cached:
                print(f"\n  {uri}  (resuming, {cached} cached)")
            else:
                print(f"\n  {uri}")
            # Print static header for live progress.
            print(f"    {'Progress':<20s}  {'Accuracy':<18s}  "
                  f"{'Prefill':<26s}  {'Generate'}", flush=True)

            correct_count = 0
            per_question: list[dict] = []
            model_info: dict = {}
            errors = 0
            total_input_tokens = cached_stats["total_input_tokens"]
            total_output_tokens = cached_stats["total_output_tokens"]

            # Throughput tracking — generate (seed with cached data).
            all_gen_toks: list[float] = list(cached_stats["gen_tok_s"])
            recent_gen_toks: deque[float] = deque(maxlen=10)
            last_gen_toks: float = 0.0
            # Throughput tracking — prefill (seed with cached data).
            all_prefill_toks: list[float] = list(cached_stats["prefill_tok_s"])
            recent_prefill_toks: deque[float] = deque(maxlen=10)
            last_prefill_toks: float = 0.0

            for i, sample in enumerate(samples):
                if (uri, sample["index"]) in completed:
                    continue

                # Build request body (scenario-specific).
                body = build_body(sample, uri, config)

                # Retry loop.
                events: list[dict] = []
                for attempt in range(_MAX_RETRIES):
                    try:
                        events, _ = http_post_stream(url, body, timeout=180)
                        if events:
                            break
                    except Exception as exc:
                        if attempt < _MAX_RETRIES - 1:
                            wait = 2 ** attempt
                            print(f"\n    retry {attempt+1} after error: {exc}", flush=True)
                            time.sleep(wait)
                        else:
                            print(f"\n    failed after {_MAX_RETRIES} retries: {exc}", flush=True)
                            errors += 1

                raw = _extract_text(events)
                predicted = extract_answer(raw)
                is_correct = predicted == sample["correct"]
                if is_correct:
                    correct_count += 1

                per_question.append({
                    "index": sample["index"],
                    "question_hash": sample["question_hash"],
                    "predicted": predicted,
                    "correct": sample["correct"],
                    "match": is_correct,
                    "raw_output": raw[:500],
                })

                if not model_info:
                    model_info = extract_generation_metrics(events).get("model_info", {})

                # Throughput from this request.
                metrics = extract_generation_metrics(events)

                logger.log(
                    bench=bench_name,
                    index=sample["index"],
                    question_hash=sample["question_hash"],
                    predicted=predicted,
                    correct=sample["correct"],
                    model=uri,
                    raw_output=raw,
                    input_tokens=metrics.get("input_tokens", 0),
                    output_tokens=metrics.get("output_tokens", 0),
                    prefill_tok_s=metrics.get("prefill_tok_s", 0),
                    gen_tok_s=metrics.get("engine_tok_s", 0),
                )
                gen_ts = metrics.get("engine_tok_s", 0)
                if gen_ts > 0:
                    all_gen_toks.append(gen_ts)
                    recent_gen_toks.append(gen_ts)
                    last_gen_toks = gen_ts
                pre_ts = metrics.get("prefill_tok_s", 0)
                if pre_ts > 0:
                    all_prefill_toks.append(pre_ts)
                    recent_prefill_toks.append(pre_ts)
                    last_prefill_toks = pre_ts

                # Token counts.
                total_input_tokens += metrics.get("input_tokens", 0)
                total_output_tokens += metrics.get("output_tokens", 0)

                # Progress line.
                done = cached + len(per_question)
                total_correct = cached_correct + correct_count
                pct = total_correct / done * 100 if done > 0 else 0

                progress = f"{done}/{total}"
                acc = f"{total_correct}/{done} ({pct:.1f}%)"

                if all_prefill_toks:
                    pp_avg = sum(all_prefill_toks) / len(all_prefill_toks)
                    pp_str = f"{total_input_tokens:,} tok  {pp_avg:.0f} t/s"
                else:
                    pp_str = "—"
                if all_gen_toks:
                    tg_avg = sum(all_gen_toks) / len(all_gen_toks)
                    tg_str = f"{total_output_tokens:,} tok  {tg_avg:.0f} t/s"
                else:
                    tg_str = "—"

                print(
                    f"\r    {progress:<20s}  {acc:<18s}  {pp_str:<26s}  {tg_str}   ",
                    end="", flush=True,
                )

            logger.close()
            print()
            evaluated = cached + len(per_question)
            total_correct = cached_correct + correct_count
            accuracy = total_correct / evaluated * 100 if evaluated > 0 else 0

            # Aggregate throughput.
            avg_gen = sum(all_gen_toks) / len(all_gen_toks) if all_gen_toks else 0
            avg_prefill = sum(all_prefill_toks) / len(all_prefill_toks) if all_prefill_toks else 0

            all_results.append({
                "model": base_model,
                "scheme": scheme,
                "model_uri": uri,
                "correct_count": total_correct,
                "total": total,
                "accuracy": accuracy,
                "model_info": model_info,
                "results": per_question,
                "bench": bench_name,
                "errors": errors,
                "avg_gen_tok_s": round(avg_gen, 1),
                "avg_prefill_tok_s": round(avg_prefill, 1),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            })

    return all_results
