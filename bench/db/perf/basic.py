"""Initial DB performance scenarios for /v1/db endpoints."""

from __future__ import annotations

import base64
import time

from scenario import Scenario

from .common import request_json, request_raw, run_load


class _DbPerfScenario(Scenario):
    """Base class for DB perf scenarios."""

    report_type = "db"
    requires_storage = True
    uses_model_matrix = False


def _round_row(op: str, round_idx: int, concurrency: int, metrics: dict) -> dict:
    row = {
        "op": op,
        "round": round_idx,
        "concurrency": concurrency,
        "model": "n/a",
        "scheme": "original",
    }
    row.update(metrics)
    return row


class SqlSelect1(_DbPerfScenario):
    name = "db/perf/sql_select1"
    family = "sql"
    description = "SQL query overhead on POST /v1/db/sql/query (SELECT 1)."
    endpoint = "POST /v1/db/sql/query"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 200))
        concurrency = int(config.get("concurrency", 8))
        results: list[dict] = []

        for i in range(rounds):
            print(f"    round {i+1}/{rounds} ...", end="", flush=True)
            metrics = run_load(
                requests,
                concurrency,
                lambda _idx: request_json(
                    base_url,
                    "POST",
                    "/v1/db/sql/query",
                    {"query": "SELECT 1 AS v"},
                ),
            )
            results.append(_round_row("sql_select1", i + 1, concurrency, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class KvBatch(_DbPerfScenario):
    name = "db/perf/kv_batch"
    family = "kv"
    description = "KV batch write throughput on POST /v1/db/kv/namespaces/{ns}/batch."
    endpoint = "POST /v1/db/kv/namespaces/{ns}/batch"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 4))
        batch_size = int(config.get("batch_size", 100))
        namespace = f"bench_kv_{int(time.time() * 1000)}"
        path = f"/v1/db/kv/namespaces/{namespace}/batch"
        results: list[dict] = []

        def make_body(req_idx: int) -> dict:
            entries = []
            for j in range(batch_size):
                key = f"k_{req_idx}_{j}"
                value = f"v-{req_idx}-{j}".encode()
                entries.append(
                    {
                        "key": key,
                        "value_base64": base64.b64encode(value).decode(),
                        "durability": "batched",
                    }
                )
            return {"entries": entries}

        for i in range(rounds):
            print(f"    round {i+1}/{rounds} ...", end="", flush=True)
            metrics = run_load(
                requests,
                concurrency,
                lambda idx: request_json(base_url, "POST", path, make_body(idx)),
            )
            row = _round_row("kv_batch", i + 1, concurrency, metrics)
            row["batch_size"] = batch_size
            results.append(row)
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class RowsWrite(_DbPerfScenario):
    name = "db/perf/rows_write"
    family = "rows"
    description = "Row write throughput on POST /v1/db/tables/{ns}/rows."
    endpoint = "POST /v1/db/tables/{ns}/rows"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 120))
        concurrency = int(config.get("concurrency", 4))
        results: list[dict] = []

        for i in range(rounds):
            namespace = f"bench_rows_w_{int(time.time() * 1000)}_{i}"
            path = f"/v1/db/tables/{namespace}/rows"
            print(f"    round {i+1}/{rounds} ...", end="", flush=True)

            def _request(req_idx: int) -> tuple[int, float]:
                key = i * 10_000_000 + req_idx + 1
                now_ms = int(time.time() * 1000)
                body = {
                    "schema_id": 10,
                    "columns": [
                        {"column_id": 1, "type": "scalar_u64", "value": key},
                        {"column_id": 2, "type": "scalar_i64", "value": now_ms},
                        {"column_id": 20, "type": "string", "value": f"payload-{key}"},
                    ],
                }
                return request_json(base_url, "POST", path, body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("rows_write", i + 1, concurrency, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class RowsScan(_DbPerfScenario):
    name = "db/perf/rows_scan"
    family = "rows"
    description = "Row scan latency on GET /v1/db/tables/{ns}/rows?schema_id=..."
    endpoint = "GET /v1/db/tables/{ns}/rows"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 120))
        concurrency = int(config.get("concurrency", 8))
        seed_rows = int(config.get("seed_rows", 500))
        scan_limit = int(config.get("scan_limit", 100))
        results: list[dict] = []

        for i in range(rounds):
            namespace = f"bench_rows_s_{int(time.time() * 1000)}_{i}"
            write_path = f"/v1/db/tables/{namespace}/rows"
            scan_path = f"/v1/db/tables/{namespace}/rows?schema_id=10&limit={scan_limit}"

            for seed_idx in range(seed_rows):
                key = seed_idx + 1
                body = {
                    "schema_id": 10,
                    "columns": [
                        {"column_id": 1, "type": "scalar_u64", "value": key},
                        {"column_id": 2, "type": "scalar_i64", "value": 1_000_000 + key},
                        {"column_id": 20, "type": "string", "value": f"seed-{key}"},
                    ],
                }
                status, _ = request_json(base_url, "POST", write_path, body)
                if status < 200 or status >= 300:
                    raise RuntimeError(f"seed write failed with HTTP {status}")

            print(f"    round {i+1}/{rounds} ...", end="", flush=True)
            metrics = run_load(
                requests,
                concurrency,
                lambda _idx: request_raw(base_url, "GET", scan_path),
            )
            row = _round_row("rows_scan", i + 1, concurrency, metrics)
            row["seed_rows"] = seed_rows
            row["scan_limit"] = scan_limit
            results.append(row)
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results
