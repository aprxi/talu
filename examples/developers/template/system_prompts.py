"""
System Prompts - Create reusable personas with partial().

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

System prompts define the model's role, constraints, and output format.
Use partial() to create reusable personas for different use cases.

Related:
- examples/basics/10_prompt_templates.py
"""

import talu

# Base template with system prompt slot
assistant = talu.PromptTemplate("""
{{ system }}

{% if context %}
Context: {{ context }}

{% endif %}
User: {{ message }}
Assistant:""")

# Create specialized personas with partial()
coder = assistant.partial(system="""You are an expert Python developer.
- Write clean, idiomatic code
- Include type hints
- Add brief comments for complex logic
- No unnecessary explanations""")

teacher = assistant.partial(system="""You are a patient teacher.
- Explain concepts simply
- Use analogies
- Check for understanding
- Encourage questions""")

analyst = assistant.partial(system="""You are a data analyst.
- Be precise with numbers
- Cite sources when possible
- Acknowledge uncertainty
- Present findings objectively""")

# Same question, different personas
question = "Explain recursion"

print("=== Coder ===")
print(coder(message=question))

print("\n=== Teacher ===")
print(teacher(message=question))

# Personas with structured output requirements
json_assistant = assistant.partial(system="""You are an API assistant.
Always respond with valid JSON in this format:
{"answer": "...", "confidence": 0.0-1.0, "sources": [...]}""")

print("\n=== JSON Output ===")
print(json_assistant(message="What is the capital of France?"))

# Chaining partials
base = talu.PromptTemplate("{{ system }}\n{{ format }}\nUser: {{ message }}")
with_persona = base.partial(system="You are an expert.")
as_json = with_persona.partial(format="Respond in JSON.")
as_yaml = with_persona.partial(format="Respond in YAML.")

print("=== Chained Partials ===")
print(as_json(message="List HTTP methods"))
print()
print(as_yaml(message="List HTTP methods"))


# Dynamic system prompts based on context
tiered = talu.PromptTemplate("""
{% if tier == 'premium' %}You are a premium assistant. Be detailed.
{% else %}You are a helpful assistant. Be brief.
{% endif %}
User: {{ message }}
""")

print("\n=== Conditional System Prompts ===")
print(tiered(tier="premium", message="Explain recursion"))
print(tiered(tier="free", message="Explain recursion"))

"""
Topics covered:
* template.render
* persona.switch
"""
