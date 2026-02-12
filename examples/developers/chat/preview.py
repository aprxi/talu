"""Preview - See the prompt before sending.

Primary API: talu.Chat
Scope: Single

The preview_prompt() method shows exactly what will be sent to the model,
including the formatted chat template. Useful for debugging prompt formatting.
"""

import talu
from talu.router import GenerationConfig

chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are a pirate.")

# Have a conversation first
_ = chat("Ahoy!")

# Preview the formatted prompt (shows the full conversation with template applied)
print("=== Default template ===")
print(chat.preview_prompt())

# Preview with a custom chat_template override via config
custom_template = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"
config = GenerationConfig(chat_template=custom_template)

print("\n=== Custom template ===")
print(chat.preview_prompt(config=config))

# Preview without the assistant generation prompt
print("\n=== Without generation prompt ===")
print(chat.preview_prompt(add_generation_prompt=False))

"""
Topics covered:
* chat.session
* chat.send
"""
