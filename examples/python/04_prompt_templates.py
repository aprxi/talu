"""Build prompts with templates.

This example shows:
- Creating reusable prompt templates
- Using loops and conditionals in templates
- Few-shot learning patterns
"""

import talu

# Create a reusable prompt
prompt = talu.PromptTemplate("""
You are a {{ role }} assistant.

User: {{ question }}
Assistant:""")

# Generate different prompts from the same template
print(prompt(role="helpful", question="What is Python?"))
print(prompt(role="concise", question="Explain recursion"))

# Few-shot learning - teach the model by example
classifier = talu.PromptTemplate("""
Classify the sentiment:

{% for ex in examples %}
Text: {{ ex.text }}
Sentiment: {{ ex.label }}

{% endfor %}
Text: {{ query }}
Sentiment:""")

print(classifier(
    examples=[
        {"text": "I love this!", "label": "positive"},
        {"text": "Terrible experience", "label": "negative"},
    ],
    query="This exceeded expectations!",
))

# Loop over items
list_prompt = talu.PromptTemplate("""
Summarize these items:
{% for item in items %}
- {{ item }}
{% endfor %}
Summary:""")

print(list_prompt(items=["apples", "bananas", "pears"]))

# Conditional blocks
conditional = talu.PromptTemplate("""
{% if tone == "formal" %}
You are a formal assistant.
{% else %}
You are a casual assistant.
{% endif %}

User: {{ question }}
Assistant:""")

print(conditional(tone="casual", question="Explain gravity in one sentence."))

# Save a rendered prompt for reuse
rendered = prompt(role="helpful", question="Explain recursion in one sentence.")
with open("/tmp/talu_04_prompt_templates_rendered.txt", "w") as f:
    f.write(rendered)
print("Saved prompt to /tmp/talu_04_prompt_templates_rendered.txt")

