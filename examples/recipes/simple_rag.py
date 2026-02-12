"""
Simple RAG - Augment queries with local context.

Job: Build a minimal RAG pipeline with embeddings and context retrieval.
Prereqs: None
Failure mode: N/A (no external prerequisites).

Demonstrates:
- Chunking text for retrieval
- Computing cosine similarity with embeddings
- Building context-augmented prompts
- Token budget management for context

Requirements:
    pip install talu

Run:
    python examples/recipes/simple_rag.py
"""

import math

from talu import Client, Chat, Tokenizer


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def chunk_text(text: str, size: int = 120) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


client = Client("Qwen/Qwen3-0.6B")
tokenizer = Tokenizer("Qwen/Qwen3-0.6B")

# Small corpus with a couple of longer docs
corpus = [
    "Talu provides a Chat API for text generation. " * 10,
    "Tokenizers convert text into token IDs for models. " * 10,
    "Prompt templates use Jinja-style syntax to render prompts. " * 10,
    "Converters create local, optimized model formats. " * 10,
]

# Split into chunks
chunks: list[str] = []
for doc in corpus:
    chunks.extend(chunk_text(doc, size=50))

query = "How do I turn text into tokens?"

query_vec = client.embed(query)
chunk_vecs = [client.embed(chunk) for chunk in chunks]

scored = list(zip(chunks, chunk_vecs))
scored.sort(key=lambda item: cosine_similarity(query_vec, item[1]), reverse=True)

# Pick top-k chunks
k = 3
selected = [chunk for chunk, _ in scored[:k]]
context = "\n\n".join(selected)

# Trim context to a token budget
max_context_tokens = 800
context_tokens = tokenizer.count_tokens(context)
print(f"Context tokens before trim: {context_tokens}")
if context_tokens > max_context_tokens:
    tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
    context = tokenizer.decode(tokens)
    print(f"Context tokens after trim: {tokenizer.count_tokens(context)}")

chat = Chat("Qwen/Qwen3-0.6B", system="Answer using only the provided context.")
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
response = chat(prompt)
print(response)

# Show which chunks were used
print("\nContext snippets:")
for idx, chunk in enumerate(selected, 1):
    print(f"{idx}. {chunk[:120]}...")

# Alternative query
query = "How do prompt templates work?"
query_vec = client.embed(query)
scored = list(zip(chunks, chunk_vecs))
scored.sort(key=lambda item: cosine_similarity(query_vec, item[1]), reverse=True)
context = "\n\n".join([chunk for chunk, _ in scored[:k]])
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
print("\nSecond query response:")
print(chat(prompt))

client.close()

"""
Topics covered:

* embeddings.generate
* embeddings.similarity
* context.augmentation
* rag.simple

Related:

* examples/developers/chat/embeddings.py
"""
