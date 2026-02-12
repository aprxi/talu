"""
Chat reference tests (require TEST_MODEL_URI_TEXT, actual inference).

Organized by functional area:
- generation/: Basic generation, streaming, token counting, embeddings
- features/: Regenerate, schema inference, thinking mode
- robustness/: Memory leaks, thread safety, stress tests
- backends/: External API backends (vLLM, llama.cpp, OpenAI)
"""
