"""
Built-in chat template presets.

Contains template strings for common chat formats like ChatML, Llama2, etc.
Use ``PromptTemplate.from_preset(name)`` to load a preset by name.
"""

from __future__ import annotations

CHATML_TEMPLATE = """\
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

LLAMA2_TEMPLATE = """\
{{ bos_token }}\
{%- for message in messages %}
{%- if message.role == 'system' %}
<<SYS>>
{{ message.content }}
<</SYS>>

{%- elif message.role == 'user' %}
[INST] {{ message.content }} [/INST]
{%- elif message.role == 'assistant' %}
{{ message.content }}{{ eos_token }}
{%- endif %}
{%- endfor %}"""

ALPACA_TEMPLATE = """\
{%- if messages[0].role == 'system' %}
{{ messages[0].content }}

{% endif %}
### Instruction:
{% for message in messages %}
{%- if message.role == 'user' %}
{{ message.content }}
{% endif %}
{%- endfor %}
### Response:
"""

VICUNA_TEMPLATE = """\
{%- for message in messages %}
{%- if message.role == 'system' %}
{{ message.content }}

{%- elif message.role == 'user' %}
USER: {{ message.content }}
{%- elif message.role == 'assistant' %}
ASSISTANT: {{ message.content }}</s>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
ASSISTANT:{% endif %}"""

ZEPHYR_TEMPLATE = """\
{%- for message in messages %}
<|{{ message.role }}|>
{{ message.content }}</s>
{% endfor %}
{%- if add_generation_prompt %}<|assistant|>
{% endif %}"""

PRESETS: dict[str, str] = {
    "chatml": CHATML_TEMPLATE,
    "llama2": LLAMA2_TEMPLATE,
    "alpaca": ALPACA_TEMPLATE,
    "vicuna": VICUNA_TEMPLATE,
    "zephyr": ZEPHYR_TEMPLATE,
}


def preset_names() -> list[str]:
    """Return available preset names."""
    return list(PRESETS)
