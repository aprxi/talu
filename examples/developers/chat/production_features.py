"""Production Features - count_tokens, logit_bias, extra_body.

Primary API: talu.Chat, talu.GenerationConfig
Scope: Single

This example demonstrates production-ready features that enable real-world
deployment scenarios like RAG pipelines, classification, and provider integration.

Key points:
- count_tokens() provides accurate token counts including template overhead
- logit_bias enables controlled generation for classification
- extra_body is the "escape hatch" for provider-specific parameters

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


# =============================================================================
# 5. A/B Testing with Different Providers
# =============================================================================

print("\n=== A/B Testing Pattern ===")

# Base config for both providers
base_config = GenerationConfig(
    temperature=0.7,
    max_tokens=200,
    stop_sequences=["\n\n"],
)

# Provider A: vLLM with specific tuning
provider_a = base_config | GenerationConfig(
    extra_body={"best_of": 2, "presence_penalty": 0.5}
)

# Provider B: Together with different tuning
provider_b = base_config | GenerationConfig(
    extra_body={"repetition_penalty": 1.1}
)

print("A/B test configs:")
print(f"  Provider A (vLLM): {provider_a.extra_body}")
print(f"  Provider B (Together): {provider_b.extra_body}")

# In production:
# if random.random() < 0.5:
#     response = chat_vllm.send(prompt, config=provider_a)
# else:
#     response = chat_together.send(prompt, config=provider_b)


# =============================================================================
# 6. Custom Headers for Enterprise Networking
# =============================================================================

print("\n=== Custom Headers for Enterprise ===")

from talu.router import ModelSpec, OpenAICompatibleBackend

# Enterprise pattern 1: Internal proxy with custom auth
# Many enterprises route LLM traffic through internal proxies that require
# custom headers for authentication, routing, or tracing.

internal_proxy_backend = OpenAICompatibleBackend(
    base_url="https://internal-llm-proxy.corp.example.com/v1",
    api_key=None,  # Proxy handles authentication
    headers={
        "X-Proxy-Auth": "internal-service-token",
        "X-Service-Name": "ml-pipeline",
        "X-Team-ID": "data-science",
        "X-Correlation-ID": "request-12345",  # For distributed tracing
    },
)

print("Internal proxy config:")
print(f"  base_url: {internal_proxy_backend.base_url}")
print(f"  headers: {internal_proxy_backend.headers}")

# Enterprise pattern 2: OpenAI with request tracing
# Add custom headers alongside the standard OpenAI authentication.

openai_with_tracing = OpenAICompatibleBackend(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",  # Your OpenAI key
    org_id="org-123",
    headers={
        "X-Request-ID": "trace-abc-123",
        "X-Environment": "production",
    },
)

print("\nOpenAI with tracing:")
print(f"  api_key: {openai_with_tracing.api_key[:10]}...")
print(f"  org_id: {openai_with_tracing.org_id}")
print(f"  headers: {openai_with_tracing.headers}")

# Enterprise pattern 3: AWS Bedrock-style auth via SigV4 headers
# (Hypothetical - actual SigV4 signing would be done separately)

bedrock_style = OpenAICompatibleBackend(
    base_url="https://bedrock.us-east-1.amazonaws.com/v1",
    headers={
        "X-Amz-Target": "AmazonBedrock.Converse",
        "X-Amz-Content-Sha256": "...",
        "Authorization": "AWS4-HMAC-SHA256 Credential=...",
    },
)

print("\nBedrock-style config:")
print(f"  headers: {list(bedrock_style.headers.keys())}")

# Using headers with Chat
# spec = ModelSpec(
#     ref="gpt-4o",
#     backend=internal_proxy_backend,
# )
# chat = Chat(spec, system="You are a helpful assistant.")
# response = chat.send("Hello!")


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
* examples/developers/chat/custom_endpoints.py
"""
