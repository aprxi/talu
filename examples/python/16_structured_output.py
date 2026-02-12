"""Get structured JSON responses.

This example shows:
- Using dataclasses for structured output (stdlib)
- Using TypedDict for type-hinted dicts (stdlib)
- Using JSON Schema dicts directly (stdlib)
- Parsing typed structured output
- Saving structured data to files
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import TypedDict

import talu
from talu import repository
from talu.router import GenerationConfig

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")


# =============================================================================
# Option 1: dataclass (returns dataclass instance)
# =============================================================================


@dataclass
class Weather:
    summary: str
    temperature_c: float
    wind_kph: float


@dataclass
class TaskList:
    tasks: list[str]
    priority: str


@dataclass
class Contact:
    name: str
    email: str
    tags: list[str]


chat = talu.Chat(MODEL_URI, config=GenerationConfig(seed=42, max_tokens=256))

# Request 1: weather summary
response = chat.send(
    "What is the weather in Paris? Return a flat JSON object with keys: summary, temperature_c, wind_kph.",
    response_format=Weather,
)

print(response.text)
weather = response.parsed  # Returns Weather instance
print(f"Temp: {weather.temperature_c} C")

# Request 2: task list
response = chat.send(
    "Plan a weekend trip. Return a flat JSON object with keys: tasks (list of strings), priority (string).",
    response_format=TaskList,
)

print(asdict(response.parsed))

# Save the structured output
with open("/tmp/talu_16_structured_output_weather.json", "w") as f:
    f.write(response.text)

# Use structured output in a follow-up prompt
tasks = response.parsed.tasks
follow_up = "Summarize these tasks in one sentence: " + ", ".join(tasks)
print(chat(follow_up))

# Use the same schema for a different request
response = chat.send(
    "Plan a grocery run. Return a flat JSON object with keys: tasks (list of two strings), priority (string).",
    response_format=TaskList,
)
print(response.parsed.tasks)

# Convert parsed output to plain dict for storage or APIs
task_dict = asdict(response.parsed)
print(task_dict)
with open("/tmp/talu_16_structured_output_tasks.json", "w") as f:
    json.dump(task_dict, f, indent=2)

# Request 3: contact card
response = chat.send(
    "Return a flat JSON object with keys: name, email, tags. Values: Jane Doe, jane@example.com, [vip, beta].",
    response_format=Contact,
)
print(asdict(response.parsed))


# =============================================================================
# Option 2: TypedDict (returns dict with type hints)
# =============================================================================


class Movie(TypedDict):
    title: str
    year: int
    rating: float


response = chat.send(
    "Recommend a classic sci-fi movie. Return a flat JSON object with keys: title, year, rating (out of 10).",
    response_format=Movie,
)
movie = response.parsed  # Returns dict, IDE knows types
print(f"Movie: {movie['title']} ({movie['year']}) - {movie['rating']}/10")


# =============================================================================
# Option 3: JSON Schema dict (most flexible)
# =============================================================================

book_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "pages": {"type": "integer"},
    },
    "required": ["title", "author", "pages"],
}

response = chat.send(
    "Recommend a programming book. Return a flat JSON object with keys: title, author, pages.",
    response_format=book_schema,
)
book = response.parsed  # JSON schema -> returns dict directly
print(f"Book: {book['title']} by {book['author']}, {book['pages']} pages")
