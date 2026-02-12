"""Embeddings - Extract dense vector embeddings for semantic search and RAG.

Primary API: talu.Client
Scope: Single

Embeddings are useful for:
- Semantic search (find similar documents)
- RAG retrieval (retrieve relevant context)
- Document clustering and classification
"""

import math

import talu
from talu import Client


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# Basic embedding extraction
client = Client("Qwen/Qwen3-0.6B")

# Get embedding for a text
embedding = client.embed("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Embedding dimension (d_model)
dim = client.embedding_dim()
print(f"Model dimension: {dim}")

# Different pooling strategies
emb_last = client.embed("Hello", pooling="last")  # Default for decoder models
emb_mean = client.embed("Hello", pooling="mean")  # Average all positions
emb_first = client.embed("Hello", pooling="first")  # First token (CLS)

print(f"\nPooling strategies produce different embeddings:")
print(f"  last vs mean: {cosine_similarity(emb_last, emb_mean):.4f}")
print(f"  last vs first: {cosine_similarity(emb_last, emb_first):.4f}")

# Normalized vs unnormalized
emb_norm = client.embed("Hello", normalize=True)  # L2 normalized (default)
emb_raw = client.embed("Hello", normalize=False)  # Raw hidden states

norm_normalized = math.sqrt(sum(x * x for x in emb_norm))
norm_raw = math.sqrt(sum(x * x for x in emb_raw))
print(f"\nNormalization:")
print(f"  Normalized L2 norm: {norm_normalized:.4f} (should be ~1.0)")
print(f"  Unnormalized L2 norm: {norm_raw:.4f}")

# Semantic similarity
print("\nSemantic similarity:")
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Python is a programming language",
]
embeddings = [client.embed(t) for t in texts]

for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{text1[:30]}...' <-> '{text2[:30]}...'")
            print(f"    Similarity: {sim:.4f}")

# Batch embedding
print("\nBatch embedding:")
batch_embeddings = client.embed_batch(texts)
print(f"  Got {len(batch_embeddings)} embeddings")
print(f"  Each has dimension {len(batch_embeddings[0])}")

client.close()

"""
Topics covered:
* embeddings.generate
* embeddings.similarity
"""
