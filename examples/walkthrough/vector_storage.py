"""TaluDB vector storage walkthrough (user-facing).

Mental model
------------
TaluDB is a high-performance *file format*, not a server. A folder is your
vector database. Point VectorStore at a path and it creates or reuses the files
inside that folder.

Persistence is automatic:
- append_batch() writes to the WAL immediately (crash-safe)
- data is flushed into current.talu automatically as buffers fill
- reopening the same folder restores the data
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

from talu.db import VectorStore


# ---------------------------------------------------------------------------
# 1) Create/Open a persistent store
# ---------------------------------------------------------------------------
# A folder is your database. If it doesn't exist, it is created.
# If it does exist, TaluDB loads the data inside it.
#
# store = VectorStore("./my-knowledge-base")
# ---------------------------------------------------------------------------


def _seed(store: VectorStore) -> None:
    """Insert three unit vectors with ids 1,2,3."""
    ids = array("Q", [1, 2, 3])  # u64 IDs
    vectors = array(
        "f",
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )
    store.append_batch(ids, vectors, dims=3)


# ---------------------------------------------------------------------------
# 2) Add data efficiently (batch append)
# ---------------------------------------------------------------------------
# TaluDB prefers batches (Structure-of-Arrays). You supply:
# - ids: array("Q", ...)  (u64)
# - vectors: array("f", ...)  (float32, flattened)
# - dims: vector dimensionality
#
# Example for 3 vectors of dim=2:
#   vectors = [v0x, v0y, v1x, v1y, v2x, v2y]
# ---------------------------------------------------------------------------


def demo_search(store: VectorStore) -> None:
    query = array("f", [1.0, 0.0, 0.0])
    ids, scores = store.search(query, k=2)
    print("search ->", list(zip(ids, scores, strict=False)))


# ---------------------------------------------------------------------------
# 3) Inspect data programmatically
# ---------------------------------------------------------------------------
# load() returns all vectors and ids. Useful for inspection/debugging.
#
# batch = store.load()
# print(batch.count, batch.ids[0])
# ---------------------------------------------------------------------------


def demo_search_batch(store: VectorStore) -> None:
    queries = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ids, scores, count = store.search_batch(queries, dims=3, query_count=2, k=2)
    print("search_batch -> count_per_query", count)
    for idx in range(2):
        offset = idx * count
        print("  q", idx, list(zip(ids[offset : offset + count], scores[offset : offset + count], strict=False)))


# ---------------------------------------------------------------------------
# 4) Stream scores without top-k sorting
# ---------------------------------------------------------------------------
# scan() yields (id, score) for every vector. This is useful if you want to
# apply custom thresholds/filters in Python.
# ---------------------------------------------------------------------------


def demo_scan(store: VectorStore) -> None:
    query = array("f", [1.0, 0.0, 0.0])
    print("scan ->", list(store.scan(query)))


# ---------------------------------------------------------------------------
# 5) Score-only batched scan into buffers (no heap)
# ---------------------------------------------------------------------------
# scan_batch() runs all queries in a single pass and returns flat buffers.
# The scores are laid out as [query][row].
# ---------------------------------------------------------------------------


def demo_scan_batch(store: VectorStore) -> None:
    queries = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ids, scores, total_rows = store.scan_batch(queries, dims=3, query_count=2)
    print("scan_batch -> total_rows", total_rows)
    for idx in range(2):
        offset = idx * total_rows
        print("  q", idx, list(zip(ids, scores[offset : offset + total_rows], strict=False)))


def main() -> None:
    parser = argparse.ArgumentParser(description="TaluDB vector storage walkthrough.")
    parser.add_argument("--db", type=Path, default=Path("./vector-demo"))
    parser.add_argument(
        "--mode",
        choices=["seed", "search", "search_batch", "scan", "scan_batch", "demo"],
        default="demo",
    )
    args = parser.parse_args()

    args.db.mkdir(parents=True, exist_ok=True)
    store = VectorStore(args.db)

    if args.mode in {"seed", "demo", "search", "search_batch", "scan", "scan_batch"}:
        _seed(store)

    if args.mode == "seed":
        print("Seeded demo vectors.")
        return
    if args.mode == "search":
        demo_search(store)
    elif args.mode == "search_batch":
        demo_search_batch(store)
    elif args.mode == "scan":
        demo_scan(store)
    elif args.mode == "scan_batch":
        demo_scan_batch(store)
    else:
        demo_search(store)
        demo_search_batch(store)
        demo_scan(store)
        demo_scan_batch(store)


if __name__ == "__main__":
    main()
