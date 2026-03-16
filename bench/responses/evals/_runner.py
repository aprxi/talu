"""Shared eval runner loop with retry, resume, and throughput tracking."""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from collections import deque

from scenario import extract_generation_metrics, model_uri
from extract import extract_answer
from log import eval_log_path, load_completed, EvalLogger

_MAX_RETRIES = 3


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

        events = _parse_sse(raw)
        return events, wall_s

    def close(self) -> None:
        self._reset()


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
    model_uris: list[str] = config.get("model_uri", ["Qwen/Qwen3.5-0.8B"])
    precisions: list[str] = config.get("precision", ["original"])
    samples_n: int | None = config.get("samples")
    if isinstance(samples_n, str):
        samples_n = int(samples_n)
    total = len(samples)

    client = _PersistentClient(base_url)
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

                # Retry loop with persistent connection.
                events: list[dict] = []
                for attempt in range(_MAX_RETRIES):
                    try:
                        events, _ = client.post("/v1/responses", body)
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
                reasoning = _extract_reasoning(events)
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
                    reasoning=reasoning,
                    question=sample.get("prompt", ""),
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

    client.close()
    return all_results
