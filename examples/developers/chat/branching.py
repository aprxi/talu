"""Branching - Explore alternative conversation paths.

Primary API: talu.Chat
Scope: Single

append() automatically forks when the conversation has moved past the response
you're appending to. This enables intuitive branching without manual fork() calls.
For explicit control, use chat.fork() to create independent copies.
"""

import talu

chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are a helpful chef.")

# =============================================================================
# Auto-Fork with append() - The Simple Way
# =============================================================================
r1 = chat("I want to cook dinner")
print(f"r1: {r1}")

# Linear append - continues on same chat
r2 = r1.append("I have chicken and vegetables")
print(f"r2: {r2}")
print(f"  (same chat: {r2.chat is chat})")

# Branch from r1 - auto-forks because chat moved past r1
asian = r1.append("Make it Asian style")
print(f"asian: {asian}")
print(f"  (forked: {asian.chat is not chat})")

# Another branch from r1
italian = r1.append("Make it Italian style")
print(f"italian: {italian}")
print(
    f"  (forked: {italian.chat is not chat}, different from asian: {italian.chat is not asian.chat})"
)

# Original chat unchanged (has the "chicken and vegetables" branch)
print(f"\nOriginal chat has {len(chat.items)} items")
print(f"Asian branch has {len(asian.chat.items)} items")
print(f"Italian branch has {len(italian.chat.items)} items")


# =============================================================================
# Explicit fork() - For More Control
# =============================================================================
# You can still use fork() explicitly when needed
base_chat = talu.Chat("Qwen/Qwen3-0.6B")
base_chat("What's a good breakfast?")

# Create explicit forks
healthy = base_chat.fork()
indulgent = base_chat.fork()

healthy("Make it healthy")
indulgent("Make it indulgent")

print(f"\nExplicit forks:")
print(f"Base chat: {len(base_chat.items)} items")
print(f"Healthy fork: {len(healthy.items)} items")
print(f"Indulgent fork: {len(indulgent.items)} items")

"""
Topics covered:
* chat.branching
* chat.fork
"""
