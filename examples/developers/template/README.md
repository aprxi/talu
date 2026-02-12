# Prompt Template Examples

For first steps, see `examples/basics/README.md`.

Build prompts for LLMs using Jinja2 syntax. The familiar syntax means no new DSL to learn, while the native Zig backend ensures fast rendering without Python's jinja2 dependency.

## Prompt Engineering Patterns

| Example | What you'll learn |
|---------|-------------------|
| `few_shot.py` | Teach the model with examples |
| `rag.py` | Retrieval-Augmented Generation prompts |
| `tool_use.py` | Format function/tool definitions |
| `chain_of_thought.py` | Guide the model's reasoning |

## Production Workflows

| Example | What you'll learn |
|---------|-------------------|
| `system_prompts.py` | Control behavior with personas |
| `validation.py` | Catch prompt errors before API calls |
| `debugging.py` | Diagnose prompt issues |

## Advanced Features

| Example | What you'll learn |
|---------|-------------------|
| `function_docs.py` | Extract tool descriptions from Python source |
| `prompt_logging.py` | Track which variables populated each prompt |
| `custom_filters.py` | Create custom filters for domain-specific formatting |
| `jinja_composition.py` | Build modular templates with `{% include %}` |
| `environment.py` | Share filters and globals across templates |
| `structured_data.py` | Generate JSON schemas for structured output |

## Chat Templates

| Example | What you'll learn |
|---------|-------------------|
| `chat_formats.py` | Format conversations for different models |
| `chat_templates.py` | Load prompt templates from model configs |
