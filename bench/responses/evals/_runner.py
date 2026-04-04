"""Shared eval runner loop with retry, resume, and throughput tracking."""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from scenario import extract_generation_metrics, model_uri
from extract import extract_answer
from log import eval_log_path, load_completed, EvalLogger
from responses.evals._api import format_request, extract_output

_MAX_RETRIES = 3
_MASK_U64 = (1 << 64) - 1
_MASK_I64_POS = (1 << 63) - 1
_GOLDEN64 = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """Deterministic 64-bit mixer (SplitMix64 finalizer)."""
    z = (value + _GOLDEN64) & _MASK_U64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK_U64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK_U64
    return (z ^ (z >> 31)) & _MASK_U64


def _derive_request_seed(base_seed: int, sample: dict) -> int:
    """Derive a stable per-sample seed from the run-level seed."""
    idx = int(sample.get("index", 0))
    qhash = sample.get("question_hash", "")
    qhash_u64 = 0
    if isinstance(qhash, str) and qhash:
        try:
            qhash_u64 = int(qhash[:16], 16) & _MASK_U64
        except ValueError:
            qhash_u64 = 0
    mixed = (int(base_seed) & _MASK_U64) ^ ((idx + 1) * _GOLDEN64 & _MASK_U64) ^ qhash_u64
    derived = _splitmix64(mixed) & _MASK_I64_POS
    # Keep endpoint-compatible range: 1..2^63-1.
    # Core treats seed=0 as non-deterministic, so avoid 0.
    return derived if derived != 0 else 1


def _extract_response_output(events: list[dict]) -> list[dict]:
    """Extract the full output array from the terminal response event."""
    for ev in events:
        if ev.get("event") in ("response.completed", "response.incomplete"):
            resp = ev.get("data", {}).get("response", {})
            return resp.get("output", [])
    return []


def _extract_text(events: list[dict]) -> str:
    """Extract generated answer text from response events (excludes reasoning)."""
    for ev in events:
        if ev["event"] in ("response.completed", "response.incomplete"):
            resp = ev["data"].get("response", {})
            for item in resp.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") in ("output_text", "text"):
                        return part.get("text", "").strip()
    return ""


def _extract_reasoning(events: list[dict]) -> str:
    """Extract thinking/reasoning text from response events."""
    for ev in events:
        if ev["event"] in ("response.completed", "response.incomplete"):
            resp = ev["data"].get("response", {})
            for item in resp.get("output", []):
                if item.get("type") == "reasoning":
                    # Reasoning summary parts.
                    for s in item.get("summary", []):
                        if s.get("type") == "summary_text":
                            return s.get("text", "")
                    # Direct content (reasoning_text parts).
                    for part in item.get("content", []):
                        if part.get("type") in ("reasoning_text", "text"):
                            return part.get("text", "")
    return ""


def _parse_sse(body_str: str) -> list[dict]:
    """Parse SSE event stream or plain JSON into event list."""
    events: list[dict] = []
    current_event = ""
    for line in body_str.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                parsed_data = json.loads(line[6:])
                events.append({"event": current_event, "data": parsed_data})
            except json.JSONDecodeError:
                pass
        elif line == "":
            current_event = ""
    # Non-streaming: body is a single JSON response.
    if not events and body_str.strip():
        try:
            resp = json.loads(body_str.strip())
            status = resp.get("status", "completed")
            ev_type = "response.completed" if status == "completed" else "response.incomplete"
            events.append({"event": ev_type, "data": {"response": resp}})
        except json.JSONDecodeError:
            pass
    return events


class _PersistentClient:
    """HTTP/1.1 keep-alive client. One TCP connection reused across requests."""

    def __init__(self, base_url: str, timeout: float = 180) -> None:
        parsed = urllib.parse.urlparse(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout,
            )
        return self._conn

    def _reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def post(self, path: str, body: dict) -> tuple[list[dict], float]:
        """POST JSON, return (events, wall_seconds). Reconnects on failure."""
        payload = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        }
        t0 = time.monotonic()
        conn = self._connect()
        try:
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode(errors="replace")
        except Exception:
            self._reset()
            raise
        wall_s = time.monotonic() - t0

        if resp.status != 200:
            print(f"\n    HTTP {resp.status}: {raw[:200]}", flush=True)
            raise RuntimeError(f"HTTP {resp.status}: {raw[:200]}")

        events = _parse_sse(raw)
        return events, wall_s

    def close(self) -> None:
        self._reset()


class _ThreadLocalPostClient:
    """One persistent HTTP client per worker thread."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._local = threading.local()
        self._clients: list[_PersistentClient] = []
        self._lock = threading.Lock()

    def post(self, path: str, body: dict) -> tuple[list[dict], float]:
        client = getattr(self._local, "client", None)
        if client is None:
            client = _PersistentClient(self._base_url)
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client.post(path, body)

    def close(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.close()


def _request_with_retries(post_fn, api_path: str, body: dict) -> tuple[list[dict], bool]:
    """Execute a request with retry/backoff and return (events, failed)."""
    events: list[dict] = []
    for attempt in range(_MAX_RETRIES):
        try:
            events, _ = post_fn(api_path, body)
            if events:
                return events, False
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return events, True


def run_eval(
    *,
    bench_name: str,
    base_url: str,
    config: dict,
    samples: list[dict],
    build_body,
    score_fn=None,
    completions: bool = False,
) -> list[dict]:
    """Run an evaluation loop over samples for each model × precision × reasoning budget.

    Args:
        bench_name: Benchmark identifier (e.g. "mmlu").
        base_url: Server base URL.
        config: Loaded config dict.
        samples: List of sample dicts with at least: prompt, correct, question_hash, index.
        build_body: Callable(sample, uri, config) -> canonical request dict.
                    Must return dict with: model, system, input, and config params.
        score_fn: Optional Callable(raw_text, sample, events) -> dict.
                  When provided, replaces the default extract_answer + equality check.
                  Must return {"predicted": str, "is_correct": bool, ...extra}.
                  Extra keys are merged into per-question results.
                  *events* is the parsed SSE event list (for extracting tool calls etc).
        completions: If True, use v1/chat/completions API format.
                     If False (default), use v1/responses API format.

    Returns:
        List of result dicts (one per model × precision × reasoning budget).
    """
    model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
    precisions: list[str] = config.get("precision", ["original"])
    mrt_values: list[int] = config.get("max_reasoning_tokens", [0])
    if not isinstance(mrt_values, list):
        mrt_values = [int(mrt_values)]
    samples_n: int | None = config.get("samples")
    if isinstance(samples_n, str):
        samples_n = int(samples_n)
    batched = max(1, int(config.get("batched", 1)))
    total = len(samples)

    client = _PersistentClient(base_url)
    all_results: list[dict] = []

    for base_model in model_uris:
        for scheme in precisions:
            is_original = scheme == "original"
            uri = model_uri(base_model, None if is_original else scheme)

            for mrt in mrt_values:
                # Set current reasoning budget in config for build_body.
                config["max_reasoning_tokens"] = mrt

                # Resume support.
                endpoint_url = base_url if completions else None
                log_path = eval_log_path(
                    bench_name,
                    uri,
                    samples_n,
                    mrt,
                    endpoint=endpoint_url,
                    session_id=config.get("_session_id"),
                )
                completed, cached_stats = load_completed(log_path)
                logger = EvalLogger(log_path)
                cached = sum(1 for (m, _) in completed if m == uri)
                cached_correct = cached_stats["cached_correct"]
                mrt_label = f"  r={mrt}" if mrt > 0 else ""
                if cached:
                    print(f"\n  {uri}{mrt_label}  (resuming, {cached} cached)")
                else:
                    print(f"\n  {uri}{mrt_label}")
                # Print static header for live progress.
                print(f"    {'Progress':<20s}  {'Accuracy':<18s}  "
                      f"{'Prefill':<26s}  {'Generate'}", flush=True)

                correct_count = 0
                per_question: list[dict] = []
                cached_meta = cached_stats.get("meta", {})
                model_info: dict = cached_meta.get("model_info", {})
                errors = 0
                total_input_tokens = cached_stats["total_input_tokens"]
                total_output_tokens = cached_stats["total_output_tokens"]

                # Throughput tracking — generate (seed with cached data).
                all_gen_toks: list[float] = list(cached_stats["gen_tok_s"])
                # Throughput tracking — prefill (seed with cached data).
                all_prefill_toks: list[float] = list(cached_stats["prefill_tok_s"])
                pending: list[tuple[dict, str, dict]] = []
                for sample in samples:
                    if (uri, sample["index"]) in completed:
                        continue

                    # Build canonical request, then format for target API.
                    canonical = build_body(sample, uri, config)
                    # Eval safety: avoid reusing the exact same seed for every question.
                    # Repeated same-seed + single-token decoding can create strong seed lottery
                    # effects. Derive a deterministic per-sample request seed instead.
                    if "seed" in canonical:
                        try:
                            base_seed = int(canonical["seed"])
                        except (TypeError, ValueError):
                            base_seed = 0
                        if base_seed != 0:
                            canonical["seed"] = _derive_request_seed(base_seed, sample)
                    api_path, body = format_request(canonical, completions=completions)
                    pending.append((sample, api_path, body))

                def _handle_result(sample: dict, events: list[dict]) -> None:
                    nonlocal correct_count, model_info, total_input_tokens, total_output_tokens

                    output = extract_output(events, completions=completions)
                    raw = output["raw_output"]
                    reasoning = output["reasoning"]

                    if score_fn:
                        score = score_fn(raw, sample, events)
                        predicted = score.get("predicted", "")
                        is_correct = score.get("is_correct", False)
                        extra = {k: v for k, v in score.items()
                                 if k not in ("predicted", "is_correct")}
                    else:
                        predicted = extract_answer(raw)
                        is_correct = predicted == sample["correct"]
                        extra = {}

                    if is_correct:
                        correct_count += 1

                    per_question.append({
                        "index": sample["index"],
                        "question_hash": sample["question_hash"],
                        "predicted": predicted,
                        "correct": sample["correct"],
                        "match": is_correct,
                        "raw_output": raw[:500],
                        **extra,
                    })

                    if not model_info:
                        model_info = extract_generation_metrics(events).get("model_info", {})
                        if model_info and not cached_meta:
                            logger.write_meta(model_info=model_info, max_reasoning_tokens=mrt)

                    # Throughput from this request.
                    metrics = extract_generation_metrics(events)

                    response_output = output.get("response_output", [])

                    logger.log(
                        bench=bench_name,
                        index=sample["index"],
                        question_hash=sample["question_hash"],
                        predicted=predicted,
                        correct=sample["correct"],
                        model=uri,
                        raw_output=raw,
                        reasoning=reasoning,
                        response_output=response_output,
                        question=sample.get("prompt", ""),
                        input_tokens=metrics.get("input_tokens", 0),
                        output_tokens=metrics.get("output_tokens", 0),
                        prefill_tok_s=metrics.get("prefill_tok_s", 0),
                        gen_tok_s=metrics.get("engine_tok_s", 0),
                        match=is_correct,
                        extras=extra if extra else None,
                    )
                    gen_ts = metrics.get("engine_tok_s", 0)
                    if gen_ts > 0:
                        all_gen_toks.append(gen_ts)
                    pre_ts = metrics.get("prefill_tok_s", 0)
                    if pre_ts > 0:
                        all_prefill_toks.append(pre_ts)

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

                if batched <= 1 or len(pending) <= 1:
                    for sample, api_path, body in pending:
                        events, failed = _request_with_retries(client.post, api_path, body)
                        if failed:
                            errors += 1
                        _handle_result(sample, events)
                else:
                    workers = min(batched, len(pending))
                    post_client = _ThreadLocalPostClient(base_url)
                    try:
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            inflight: dict[Future, tuple[dict, str, dict]] = {}
                            next_idx = 0

                            def _submit(idx: int) -> None:
                                sample, api_path, body = pending[idx]
                                fut = pool.submit(_request_with_retries, post_client.post, api_path, body)
                                inflight[fut] = (sample, api_path, body)

                            while next_idx < len(pending) and len(inflight) < workers:
                                _submit(next_idx)
                                next_idx += 1

                            while inflight:
                                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                                for fut in done:
                                    sample, _, _ = inflight.pop(fut)
                                    try:
                                        events, failed = fut.result()
                                    except Exception:
                                        events, failed = [], True
                                    if failed:
                                        errors += 1
                                    _handle_result(sample, events)
                                    if next_idx < len(pending):
                                        _submit(next_idx)
                                        next_idx += 1
                    finally:
                        post_client.close()

                logger.close()
                print()
                evaluated = cached + len(per_question)
                total_correct = cached_correct + correct_count
                accuracy = total_correct / evaluated * 100 if evaluated > 0 else 0

                # Aggregate throughput.
                avg_gen = sum(all_gen_toks) / len(all_gen_toks) if all_gen_toks else 0
                avg_prefill = sum(all_prefill_toks) / len(all_prefill_toks) if all_prefill_toks else 0
                per_question.sort(key=lambda row: row["index"])

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
                    "max_reasoning_tokens": mrt,
                })

    client.close()
    return all_results
