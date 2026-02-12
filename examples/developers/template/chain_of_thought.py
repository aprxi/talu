"""
Chain of Thought - Use step hints and multi-path reasoning templates.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Chain-of-thought prompting improves accuracy on complex tasks by
encouraging the model to show its work before giving a final answer.

Related:
- examples/basics/10_prompt_templates.py
"""

import talu

# Basic CoT with step hints
cot = talu.PromptTemplate("""
Solve this step by step.

Problem: {{ problem }}

{% if hints %}
Hints:
{% for hint in hints %}
- {{ hint }}
{% endfor %}
{% endif %}

Think through each step, then give the final answer.
""")

print(cot(
    problem="A train travels 120 miles in 2 hours. What is its speed in miles per minute?",
    hints=["First calculate miles per hour", "Then convert hours to minutes"],
))

# Structured reasoning template
reasoning = talu.PromptTemplate("""
Analyze this problem using structured reasoning.

Problem: {{ problem }}

1. **Understand**: What is being asked?
2. **Plan**: What steps are needed?
3. **Execute**: Work through each step.
4. **Verify**: Check the answer makes sense.

Show your work:
""")

print(reasoning(problem="If 3 machines make 3 widgets in 3 minutes, how long for 100 machines to make 100 widgets?"))

# Self-consistency: generate multiple reasoning paths
multi_path = talu.PromptTemplate("""
Solve this problem {{ n_paths }} different ways, then pick the most common answer.

Problem: {{ problem }}

{% for i in range(n_paths) %}
--- Approach {{ i + 1 }} ---
{% endfor %}

Most likely answer:
""")

print(multi_path(problem="What is 17 * 24?", n_paths=3))

# Critique and refine
critique = talu.PromptTemplate("""
{% if draft %}
Here's a draft answer:
{{ draft }}

Critique this answer. Is it correct? Complete? Clear?
If not, provide an improved answer.
{% else %}
Question: {{ question }}
Answer:
{% endif %}
""")

# First pass
print(critique(question="Explain machine learning", draft=None))

# Refinement pass
print(critique(
    question="Explain machine learning",
    draft="Machine learning is when computers learn from data.",
))

"""
Topics covered:
* template.render
* chat.templates
"""
