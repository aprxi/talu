"""Session Management - Query and organize stored conversations.

Primary API: talu.Chat, talu.Database
Scope: Single

This shows how to use TaluDB to manage sessions with document lineage
tracking and advanced querying.
"""

import talu
from talu.db import Database

# Use TaluDB for persistence (requires session_id)
db = Database("talu:///tmp/example_db")

# Create a session linked to a prompt document
# source_doc_id tracks which document spawned this conversation
chat = talu.Chat(
    "Qwen/Qwen3-0.6B",
    storage=db,
    session_id="chat-001",
    source_doc_id="prompt-template-v1",  # Links to a prompt document
)
chat.append("user", "Hello!")
chat.close()

# Create another session from the same prompt
chat2 = talu.Chat(
    "Qwen/Qwen3-0.6B",
    storage=db,
    session_id="chat-002",
    source_doc_id="prompt-template-v1",
)
chat2.append("user", "Hi there!")
chat2.close()

# Create a session from a different prompt
chat3 = talu.Chat(
    "Qwen/Qwen3-0.6B",
    storage=db,
    session_id="chat-003",
    source_doc_id="prompt-template-v2",
)
chat3.append("user", "Good morning!")
chat3.close()

# Query: Find all sessions from a specific prompt (lineage query)
sessions = db.list_sessions_by_source("prompt-template-v1")
print(f"Sessions from prompt-template-v1: {len(sessions)}")
for s in sessions:
    print(f"  - {s['session_id']}")

# Query: List all sessions with various filters
all_sessions = db.list_sessions(limit=10)
print(f"\nAll sessions: {len(all_sessions)}")

# Query: Filter by source document
v2_sessions = db.list_sessions(source_doc_id="prompt-template-v2")
print(f"Sessions from v2: {len(v2_sessions)}")

"""
Topics covered:
* storage.taludb
* session.source_doc_id
* db.list_sessions
* db.list_sessions_by_source
"""
