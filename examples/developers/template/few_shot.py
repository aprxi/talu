"""
Few-Shot Learning - Use loops to include example patterns in prompts.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Few-shot prompting is one of the most effective techniques for steering
LLM behavior without fine-tuning. The model learns the pattern from
examples you provide in the prompt.

Related:
- examples/basics/10_prompt_templates.py
"""

import talu

# Sentiment classifier with examples
sentiment = talu.PromptTemplate("""
Analyze the sentiment of customer feedback.

{% for ex in examples %}
Feedback: {{ ex.text }}
Sentiment: {{ ex.sentiment }}
{% endfor %}
Feedback: {{ query }}
Sentiment:""")

# The examples teach the model your labeling style
examples = [
    {"text": "Fast shipping, great product!", "sentiment": "positive"},
    {"text": "Broke after one week.", "sentiment": "negative"},
    {"text": "It works as described.", "sentiment": "neutral"},
]

print(sentiment(examples=examples, query="Best purchase I've ever made!"))

# Entity extraction with examples
extractor = talu.PromptTemplate("""
Extract entities from the text.

{% for ex in examples %}
Text: {{ ex.text }}
Entities: {{ ex.entities | tojson }}

{% endfor %}
Text: {{ query }}
Entities:""")

examples = [
    {"text": "John works at Google in NYC", "entities": {"person": "John", "company": "Google", "location": "NYC"}},
    {"text": "Sarah from Apple called", "entities": {"person": "Sarah", "company": "Apple"}},
]

print(extractor(examples=examples, query="Meet Bob from Microsoft in Seattle"))

# Dynamic example selection - use more examples for hard cases
template = talu.PromptTemplate("""
{% if examples %}
Learn from these examples:
{% for ex in examples %}
Q: {{ ex.q }}
A: {{ ex.a }}
{% endfor %}
{% endif %}
Q: {{ question }}
A:""")

# Zero-shot (no examples)
print(template(examples=[], question="What is 2+2?"))

# Few-shot (with examples)
print(template(
    examples=[{"q": "What is 3+3?", "a": "6"}, {"q": "What is 5+5?", "a": "10"}],
    question="What is 7+7?",
))

"""
Topics covered:
* template.render
* prompt.few.shot
"""
