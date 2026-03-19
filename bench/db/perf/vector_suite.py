"""Vector DB performance scenarios."""

from __future__ import annotations

from scenario import Scenario

from .common import request_json, request_raw, run_load


class _VectorDbPerfScenario(Scenario):
    """Base class for vector DB perf scenarios."""

    family = "vector"
    report_type = "db"
    requires_storage = True
    uses_model_matrix = False


def _vector_dims(config: dict) -> int:
    dims = int(config.get("vector_dims", 8))
    if dims <= 0:
        raise ValueError("vector_dims must be > 0")
    return dims


def _collection_name(op: str, round_idx: int) -> str:
    return f"bench_vector_{op}_r{round_idx}"


def _round_row(op: str, round_idx: int, concurrency: int, dims: int, metrics: dict) -> dict:
    row = {
        "op": op,
        "round": round_idx,
        "concurrency": concurrency,
        "model": "n/a",
        "scheme": "original",
        "vector_dims": dims,
    }
    row.update(metrics)
    return row


def _vector_values(dims: int, seed: int) -> list[float]:
    values = [0.0] * dims
    values[(seed - 1) % dims] = 1.0
    return values


def _create_collection(base_url: str, name: str, dims: int) -> None:
    status, _ = request_json(
        base_url,
        "POST",
        "/v1/db/vectors/collections",
        {"name": name, "dims": dims},
    )
    if status < 200 or status >= 300:
        raise RuntimeError(f"create_collection failed with HTTP {status}")


def _seed_vectors(base_url: str, collection: str, dims: int, count: int) -> None:
    if count <= 0:
        return

    vectors = [
        {"id": idx, "values": _vector_values(dims, idx)}
        for idx in range(1, count + 1)
    ]
    status, _ = request_json(
        base_url,
        "POST",
        f"/v1/db/vectors/collections/{collection}/points/append",
        {"vectors": vectors},
    )
    if status < 200 or status >= 300:
        raise RuntimeError(f"seed append failed with HTTP {status}")


def _delete_ids(base_url: str, collection: str, ids: list[int]) -> None:
    if not ids:
        return

    status, _ = request_json(
        base_url,
        "POST",
        f"/v1/db/vectors/collections/{collection}/points/delete",
        {"ids": ids},
    )
    if status < 200 or status >= 300:
        raise RuntimeError(f"seed delete failed with HTTP {status}")


def _prepare_round_collection(
    base_url: str,
    op: str,
    round_idx: int,
    dims: int,
    *,
    seed_count: int = 0,
    delete_ids: list[int] | None = None,
) -> str:
    collection = _collection_name(op, round_idx)
    _create_collection(base_url, collection, dims)
    _seed_vectors(base_url, collection, dims, seed_count)
    _delete_ids(base_url, collection, delete_ids or [])
    return collection


class VectorCreateCollection(_VectorDbPerfScenario):
    """Benchmark collection creation requests."""

    name = "db/perf/vector_create_collection"
    family = "vector"
    description = "Collection creation throughput on POST /v1/db/vectors/collections."
    endpoint = "POST /v1/db/vectors/collections"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            name_prefix = f"{_collection_name('create', i + 1)}"

            def _request(req_idx: int) -> tuple[int, float]:
                body = {"name": f"{name_prefix}_{req_idx + 1}", "dims": dims}
                return request_json(base_url, "POST", "/v1/db/vectors/collections", body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("create_collection", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorAppendPoints(_VectorDbPerfScenario):
    """Benchmark point appends on a seeded collection."""

    name = "db/perf/vector_append_points"
    family = "vector"
    description = "Append throughput on POST /v1/db/vectors/collections/{name}/points/append."
    endpoint = "POST /v1/db/vectors/collections/{name}/points/append"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "append", i + 1, dims)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            base_id = i * requests
            path = f"/v1/db/vectors/collections/{collection}/points/append"

            def _request(req_idx: int) -> tuple[int, float]:
                point_id = base_id + req_idx + 1
                body = {
                    "vectors": [
                        {"id": point_id, "values": _vector_values(dims, point_id)}
                    ]
                }
                return request_json(base_url, "POST", path, body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("append_points", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorQueryPoints(_VectorDbPerfScenario):
    """Benchmark single-vector queries against a seeded collection."""

    name = "db/perf/vector_query_points"
    family = "vector"
    description = "Query latency on POST /v1/db/vectors/collections/{name}/points/query."
    endpoint = "POST /v1/db/vectors/collections/{name}/points/query"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "query", i + 1, dims, seed_count=1)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/points/query"
            query_vector = _vector_values(dims, 1)

            def _request(_req_idx: int) -> tuple[int, float]:
                body = {"vector": query_vector, "top_k": 1}
                return request_json(base_url, "POST", path, body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("query_points", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorFetchPoints(_VectorDbPerfScenario):
    """Benchmark point fetches against a seeded collection."""

    name = "db/perf/vector_fetch_points"
    family = "vector"
    description = "Fetch throughput on POST /v1/db/vectors/collections/{name}/points/fetch."
    endpoint = "POST /v1/db/vectors/collections/{name}/points/fetch"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "fetch", i + 1, dims, seed_count=1)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/points/fetch"

            def _request(_req_idx: int) -> tuple[int, float]:
                return request_json(base_url, "POST", path, {"ids": [1]})

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("fetch_points", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorUpsertPoints(_VectorDbPerfScenario):
    """Benchmark point upserts against a seeded collection."""

    name = "db/perf/vector_upsert_points"
    family = "vector"
    description = "Upsert throughput on POST /v1/db/vectors/collections/{name}/points/upsert."
    endpoint = "POST /v1/db/vectors/collections/{name}/points/upsert"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "upsert", i + 1, dims, seed_count=1)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/points/upsert"

            def _request(req_idx: int) -> tuple[int, float]:
                point_id = 1
                body = {
                    "vectors": [
                        {
                            "id": point_id,
                            "values": _vector_values(dims, req_idx + 1),
                        }
                    ]
                }
                return request_json(base_url, "POST", path, body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("upsert_points", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorDeletePoints(_VectorDbPerfScenario):
    """Benchmark point deletes against a seeded collection."""

    name = "db/perf/vector_delete_points"
    family = "vector"
    description = "Delete throughput on POST /v1/db/vectors/collections/{name}/points/delete."
    endpoint = "POST /v1/db/vectors/collections/{name}/points/delete"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "delete", i + 1, dims, seed_count=requests)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/points/delete"

            def _request(req_idx: int) -> tuple[int, float]:
                point_id = req_idx + 1
                return request_json(base_url, "POST", path, {"ids": [point_id]})

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("delete_points", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorStats(_VectorDbPerfScenario):
    """Benchmark collection stats reads."""

    name = "db/perf/vector_stats"
    family = "vector"
    description = "Stats latency on GET /v1/db/vectors/collections/{name}/stats."
    endpoint = "GET /v1/db/vectors/collections/{name}/stats"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "stats", i + 1, dims, seed_count=2)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/stats"

            def _request(_req_idx: int) -> tuple[int, float]:
                return request_raw(base_url, "GET", path)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("stats", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorChanges(_VectorDbPerfScenario):
    """Benchmark collection change-feed reads."""

    name = "db/perf/vector_changes"
    family = "vector"
    description = "Change-feed latency on GET /v1/db/vectors/collections/{name}/changes."
    endpoint = "GET /v1/db/vectors/collections/{name}/changes"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 8))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "changes", i + 1, dims, seed_count=2)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/changes?since=0&limit=32"

            def _request(_req_idx: int) -> tuple[int, float]:
                return request_raw(base_url, "GET", path)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("changes", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorCompact(_VectorDbPerfScenario):
    """Benchmark collection compaction."""

    name = "db/perf/vector_compact"
    family = "vector"
    description = "Compaction throughput on POST /v1/db/vectors/collections/{name}/compact."
    endpoint = "POST /v1/db/vectors/collections/{name}/compact"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 1))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(
                base_url,
                "compact",
                i + 1,
                dims,
                seed_count=2,
                delete_ids=[2],
            )
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/compact"

            def _request(_req_idx: int) -> tuple[int, float]:
                return request_json(base_url, "POST", path, {})

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("compact", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results


class VectorIndexesBuild(_VectorDbPerfScenario):
    """Benchmark index build requests."""

    name = "db/perf/vector_indexes_build"
    family = "vector"
    description = "Index build throughput on POST /v1/db/vectors/collections/{name}/indexes/build."
    endpoint = "POST /v1/db/vectors/collections/{name}/indexes/build"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        requests = int(config.get("requests", 100))
        concurrency = int(config.get("concurrency", 1))
        dims = _vector_dims(config)
        results: list[dict] = []

        for i in range(rounds):
            collection = _prepare_round_collection(base_url, "indexes_build", i + 1, dims, seed_count=8)
            print(f"    round {i + 1}/{rounds} ...", end="", flush=True)
            path = f"/v1/db/vectors/collections/{collection}/indexes/build"

            def _request(_req_idx: int) -> tuple[int, float]:
                body = {"max_segments": 8}
                return request_json(base_url, "POST", path, body)

            metrics = run_load(requests, concurrency, _request)
            results.append(_round_row("indexes_build", i + 1, concurrency, dims, metrics))
            print(f" {metrics['rps']} req/s p95={metrics['p95_ms']}ms", flush=True)

        return results
