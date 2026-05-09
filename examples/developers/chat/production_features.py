"""Production Features - count_tokens and logit_bias.

Primary API: talu.Chat, talu.GenerationConfig
Scope: Single

This example demonstrates production-ready features that enable real-world
deployment scenarios like RAG pipelines and classification.

Key points:
- count_tokens() provides accurate token counts including template overhead
- logit_bias enables controlled generation for classification

Related:
    - examples/developers/chat/configuration.py (basic config)
    - examples/developers/chat/config_composition.py (merging configs)
"""

from talu import Chat
from talu.router import GenerationConfig

# =============================================================================
# 1. Context Window Management (count_tokens)
# =============================================================================

print("=== Context Window Management ===")

chat = Chat("Qwen/Qwen3-0.6B", system="You are a helpful assistant.")

# Count current tokens (includes system prompt + template overhead)
current_tokens = chat.count_tokens()
print(f"Base context (system + template): {current_tokens} tokens")

# Simulate adding a document
document = "This is a long document about machine learning. " * 50

# Check if document fits before adding
if chat.count_tokens(document) < 2048:  # Assume 2048 context limit
    print(f"Document would use: {chat.count_tokens(document)} tokens")
    print("Document fits in context!")
else:
    print("Document too large - need to summarize or chunk")

# After sending a message, count updates
response = chat.send("What is 2+2?", max_tokens=10)
print(f"After one exchange: {chat.count_tokens()} tokens")


# =============================================================================
# 2. Controlled Generation (logit_bias)
# =============================================================================

print("\n=== Controlled Generation (logit_bias) ===")

# For classification, force the model to output specific tokens
# First, get token IDs for your target words using the tokenizer:
#   tokenizer = Tokenizer("model")
#   yes_id = tokenizer.encode("Yes")[0]
#   no_id = tokenizer.encode("No")[0]

# Example token IDs (these vary by model)
YES_TOKEN_ID = 9891
NO_TOKEN_ID = 2841

classification_config = GenerationConfig(
    temperature=0.0,  # Deterministic
    max_tokens=1,      # Only need one token
    logit_bias={
        YES_TOKEN_ID: 100.0,   # Strongly prefer
        NO_TOKEN_ID: 100.0,    # Strongly prefer
        # All other tokens implicitly get bias 0
    },
)

print("Classification config with logit_bias:")
print(f"  temperature: {classification_config.temperature}")
print(f"  max_tokens: {classification_config.max_tokens}")
print(f"  logit_bias: {classification_config.logit_bias}")

# Use for spam classification:
# chat = Chat("model")
# response = chat.send(
#     f"Is this spam? Answer Yes or No only.\n\n{email_text}",
#     config=classification_config
# )
# is_spam = "Yes" in str(response)


# =============================================================================
# 3. Provider-Specific Parameters (extra_body)
# =============================================================================

print("\n=== Provider-Specific Parameters (extra_body) ===")

# extra_body passes arbitrary JSON to remote APIs
# Useful for bleeding-edge parameters not yet in GenerationConfig

# vLLM-specific parameters
vllm_config = GenerationConfig(
    temperature=0.7,
    max_tokens=100,
    extra_body={
        "best_of": 3,              # vLLM: generate 3, return best
        "use_beam_search": False,   # vLLM: disable beam search
        "presence_penalty": 0.5,    # OpenAI-compatible
        "frequency_penalty": 0.3,   # OpenAI-compatible
    },
)

print("vLLM config with extra_body:")
print(f"  standard: temp={vllm_config.temperature}, max_tokens={vllm_config.max_tokens}")
print(f"  extra_body: {vllm_config.extra_body}")

# Together.ai-specific parameters
together_config = GenerationConfig(
    temperature=0.7,
    extra_body={
        "repetition_penalty": 1.1,        # Together-specific
        "top_k_return_sequences": 1,      # Together-specific
    },
)

print("\nTogether.ai config with extra_body:")
print(f"  extra_body: {together_config.extra_body}")

# Ollama-specific parameters
ollama_config = GenerationConfig(
    temperature=0.7,
    extra_body={
        "num_ctx": 4096,           # Ollama: context window
        "num_gpu": 1,              # Ollama: GPU layers
        "mirostat": 2,             # Ollama: Mirostat sampling
        "mirostat_eta": 0.1,
        "mirostat_tau": 5.0,
    },
)

print("\nOllama config with extra_body:")
print(f"  extra_body: {ollama_config.extra_body}")


# =============================================================================
# 4. RAG Pipeline Pattern
# =============================================================================

print("\n=== RAG Pipeline Pattern ===")

# RAG budgeting: ensure context fits before generation

def rag_generate(chat: Chat, context: str, query: str, max_context: int = 4096):
    """Generate with RAG context, respecting context limits."""
    # Reserve tokens for response
    response_budget = 512
    available = max_context - response_budget

    # Check if context fits
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    needed = chat.count_tokens(prompt)

    if needed > available:
        # Truncate context (in practice: chunk or summarize)
        print(f"Context too large ({needed} > {available}), truncating...")
        # Simple truncation - real apps would chunk/summarize
        context = context[: len(context) // 2]
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

    print(f"Final prompt: {chat.count_tokens(prompt)} tokens")
    return chat.send(prompt, max_tokens=response_budget)


# Example usage:
# context = fetch_documents(query)
# response = rag_generate(chat, context, "What is the main topic?")


"""
Topics covered:
* generation.config
* chat.context
* config.sampling
* client.ask
* chat.streaming
* workflow.end.to.end

Related:
* examples/developers/chat/configuration.py
* examples/developers/chat/config_composition.py
"""
