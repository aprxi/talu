"""Shared HTTP and metrics helpers for DB performance scenarios."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import time
import urllib.parse
from collections.abc import Callable


def request_raw(
    base_url: str,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, float]:
    """Issue one HTTP request and return (status_code, latency_ms)."""
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    req_headers = dict(headers or {})
    payload = body
    if payload is not None and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(payload))

    t0 = time.perf_counter()
    try:
        conn.request(method, path, body=payload, headers=req_headers)
        resp = conn.getresponse()
        _ = resp.read()
        status = int(resp.status)
    except Exception:
        status = 0
    finally:
        conn.close()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return status, latency_ms


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict | None,
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> tuple[int, float]:
    """Issue a JSON request and return (status_code, latency_ms)."""
    payload = json.dumps(body).encode() if body is not None else None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    return request_raw(
        base_url,
        method,
        path,
        body=payload,
        headers=req_headers,
        timeout=timeout,
    )


def run_load(
    total_requests: int,
    concurrency: int,
    request_fn: Callable[[int], tuple[int, float]],
) -> dict:
    """Run load with bounded client-side concurrency and summarize metrics."""
    if total_requests <= 0:
        raise ValueError("total_requests must be > 0")
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")

    ok = 0
    errors = 0
    latencies: list[float] = []

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_fn, i) for i in range(total_requests)]
        for fut in concurrent.futures.as_completed(futures):
            try:
                status, latency_ms = fut.result()
            except Exception:
                status = 0
                latency_ms = 0.0
            if 200 <= status < 300:
                ok += 1
            else:
                errors += 1
            if latency_ms > 0:
                latencies.append(latency_ms)
    wall_s = max(1e-9, time.perf_counter() - t0)

    latencies.sort()
    avg_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    p50 = _percentile(latencies, 50.0)
    p95 = _percentile(latencies, 95.0)
    p99 = _percentile(latencies, 99.0)

    return {
        "requests": total_requests,
        "ok": ok,
        "errors": errors,
        "rps": round(total_requests / wall_s, 1),
        "avg_ms": round(avg_ms, 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "wall_s": round(wall_s, 3),
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil((p / 100.0) * len(sorted_values)))
    idx = min(len(sorted_values) - 1, rank - 1)
    return float(sorted_values[idx])
