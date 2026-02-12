"""
Prompt Logging - Track variable usage with debug=True and result.spans.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Log prompt composition for debugging and auditing.

Related:
- examples/developers/template/debugging.py
"""

import talu

# Production prompt
rag = talu.PromptTemplate("""
Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:""")


def generate_with_logging(template, **kwargs):
    """Generate prompt and log variable usage."""
    result = template(debug=True, **kwargs)

    # Log what went into this prompt
    variables_used = {}
    for span in result.spans:
        if span.is_variable:
            variables_used[span.source] = len(span.text)

    print(f"[LOG] Variables: {variables_used}")

    # Check if any document content made it into the prompt
    if not any("doc" in s.source for s in result.spans if s.is_variable):
        print("[LOG] WARNING: no document content in prompt")

    return result.output


# Normal case
prompt = generate_with_logging(
    rag,
    documents=[{"content": "Paris is the capital of France."}],
    question="What is the capital of France?",
)

# Empty documents case
prompt = generate_with_logging(
    rag,
    documents=[],
    question="What is the capital of France?",
)

"""
Topics covered:
* template.render
* template.control.flow
"""
