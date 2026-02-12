"""
RAG - Structure retrieved documents with metadata into prompts.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

RAG is the standard pattern for grounding LLM responses in your data.
The template structures retrieved documents into a prompt that guides
the model to answer based on provided context.

Related:
- examples/basics/10_prompt_templates.py
"""

import talu

# Basic RAG template
rag = talu.PromptTemplate("""
Answer the question using only the provided context.
If the answer is not in the context, say "I don't know."

Context:
{% for doc in documents %}
[{{ doc.source }}]: {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:""")

# Simulate retrieved documents from your vector store
documents = [
    {"source": "FAQ", "content": "Returns are accepted within 30 days of purchase."},
    {"source": "Policy", "content": "Refunds are processed within 5-7 business days."},
]

print(rag(documents=documents, question="How long do refunds take?"))

# RAG with relevance scores - helps the model prioritize
scored_rag = talu.PromptTemplate("""
Use the most relevant sources to answer. Higher scores = more relevant.

{% for doc in documents %}
[Score: {{ doc.score }}] {{ doc.title }}
{{ doc.content }}

{% endfor %}
Question: {{ question }}
Answer:""")

documents = [
    {"title": "Refund Policy", "content": "Refunds are processed in 5-7 business days.", "score": 0.95},
    {"title": "Shipping FAQ", "content": "Standard shipping takes 3-5 days.", "score": 0.31},
]

print(scored_rag(documents=documents, question="When will I get my refund?"))

# Chunked documents with metadata
chunk_rag = talu.PromptTemplate("""
Answer based on the document excerpts below.

{% for chunk in chunks %}
--- {{ chunk.doc_title }} (page {{ chunk.page }}) ---
{{ chunk.text }}

{% endfor %}
Question: {{ question }}
Answer:""")

chunks = [
    {"doc_title": "User Manual", "page": 12, "text": "To reset, hold the power button for 10 seconds."},
    {"doc_title": "User Manual", "page": 15, "text": "Factory reset erases all user data."},
]

print(chunk_rag(chunks=chunks, question="How do I reset the device?"))

"""
Topics covered:
* rag.simple
* context.augmentation
"""
