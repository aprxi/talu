"""Restore Session - Initialize chat from external data.

Primary API: talu.Chat
Scope: Single

This example shows how to restore a chat session from data stored elsewhere
(Redis, database, API, etc.). This is the pattern for:
- Multi-user servers that persist sessions
- Resuming conversations across restarts
- Loading chat history from external systems

Key insight: Fork and restore are the same operation internally.
Both create a new Zig Messages handle and load data into it.

Related:
- examples/basics/08_save_restore.py
- examples/basics/18_chat_history.py
"""

import talu

# =============================================================================
# Pattern 1: Restore from saved dict (database, Redis, etc.)
# =============================================================================
# Imagine this comes from your database
saved_session = {
    "config": {
        "temperature": 0.7,
        "max_tokens": 256,
    },
    "messages": [
        {"role": "system", "content": "You are a helpful cooking assistant."},
        {"role": "user", "content": "I want to cook dinner"},
        {
            "role": "assistant",
            "content": "I'd be happy to help! What ingredients do you have?",
        },
        {"role": "user", "content": "I have chicken and vegetables"},
        {
            "role": "assistant",
            "content": "Great! There are many dishes you can make...",
        },
    ],
}

# Restore and continue the conversation
chat = talu.Chat.from_dict(saved_session, model="Qwen/Qwen3-0.6B")
# Use chat.items (Responses API) to inspect history
print(f"Restored {len(chat.items)} items")

# Continue where we left off
response = chat("Suggest something Asian")
print(f"Response: {response}")

# =============================================================================
# Pattern 2: Restore messages only (minimal data)
# =============================================================================
# Sometimes you only have the messages (e.g., from an API)
messages_only = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
]
chat = talu.Chat.from_dict({"messages": messages_only}, model="Qwen/Qwen3-0.6B")
response = chat("What are its main features?")

# =============================================================================
# Pattern 3: Multi-user server with session restore
# =============================================================================
# Simulate a database of user sessions
user_sessions_db: dict[str, dict] = {
    "user_alice": {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain calculus"},
            {"role": "assistant", "content": "Calculus is the study of change..."},
        ]
    },
    "user_bob": {
        "messages": [
            {"role": "system", "content": "You are a chef."},
            {"role": "user", "content": "How do I make pasta?"},
            {"role": "assistant", "content": "Here's a simple pasta recipe..."},
        ]
    },
}


def handle_user_message(
    user_id: str, message: str, model: str = "Qwen/Qwen3-0.6B"
) -> str:
    """Handle a message from a user, restoring their session if it exists."""
    # Load existing session or start fresh
    if user_id in user_sessions_db:
        chat = talu.Chat.from_dict(user_sessions_db[user_id], model=model)
    else:
        chat = talu.Chat(model)

    # Generate response
    response = chat(message)

    # Save updated session back to "database"
    user_sessions_db[user_id] = chat.to_dict()

    return str(response)


# Simulate requests
print(handle_user_message("user_alice", "What about integrals?"))
print(handle_user_message("user_bob", "What sauce goes with it?"))
print(handle_user_message("user_new", "Hello!"))  # New user, fresh session


# =============================================================================
# Pattern 4: Fork vs. restore (not the same)
# =============================================================================
# fork() is a high-fidelity clone (preserves timestamps, telemetry, and lineage).
# from_dict()/to_dict() use the OpenAI message format and are intentionally lossy.
original = talu.Chat("Qwen/Qwen3-0.6B", system="You are helpful.")
response = original("I have a question about Python")

# Fork using the fork() method (high-fidelity)
forked_a = original.fork()

# Restore via OpenAI-format messages (lossy interchange)
forked_b = talu.Chat.from_dict(original.to_dict(), model="Qwen/Qwen3-0.6B")

# Both chats have independent copies of the history
forked_a("Tell me about lists")
forked_b("Tell me about dictionaries")

# Original unchanged
print(f"Original: {len(original.items)} items")
print(f"Forked A: {len(forked_a.items)} items")
print(f"Forked B: {len(forked_b.items)} items")

"""
Topics covered:
* chat.persistence
* chat.history
"""
