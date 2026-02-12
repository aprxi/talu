"""Get text embeddings for similarity and search.

This example shows:
- Generating embeddings for text
- Computing cosine similarity
- Batch embedding for search applications
"""

import math
import os
import sys

from talu import Client
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


with Client(MODEL_URI) as client:
    embedding = client.embed("Hello, world!")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Model dimension: {client.embedding_dim()}")

    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "Python is a programming language",
    ]

    batch = client.embed_batch(texts)
    print(f"Batch size: {len(batch)}")

    sim = cosine_similarity(batch[0], batch[1])
    print(f"Similarity (0 vs 1): {sim:.4f}")

    query = "Cats on rugs"
    query_vec = client.embed(query)
    scored = [(text, cosine_similarity(query_vec, vec)) for text, vec in zip(texts, batch)]
    scored.sort(key=lambda item: item[1], reverse=True)
    print(f"Best match for '{query}': {scored[0][0]}")

    # Normalize vs raw embeddings
    raw = client.embed("Hello", normalize=False)
    norm = client.embed("Hello", normalize=True)
    print(f"Raw norm: {math.sqrt(sum(x * x for x in raw)):.2f}")
    print(f"Normalized norm: {math.sqrt(sum(x * x for x in norm)):.2f}")

