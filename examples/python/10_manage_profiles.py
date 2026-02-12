"""Manage persistent chat sessions with profiles.

This example shows:
- Enabling persistence with ``Profile``
- Listing and searching sessions
- Resuming a prior session by session_id
- Isolating sessions across profiles
"""

import os
import sys

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")

# Explicit dev profile
dev = talu.Profile("dev")

chat = talu.Chat(MODEL_URI, profile=dev, system="You are helpful.")
response = chat.send("What are the benefits of code review?")
print(response)
print(f"Saved session: {chat.session_id}")

saved_id = chat.session_id
assert saved_id is not None

# Resume existing session
resumed = talu.Chat(MODEL_URI, profile=dev, session_id=saved_id)
print(resumed.send("Summarize that answer in 3 bullet points."))

# List sessions in current profile
for session in dev.sessions(limit=5):
    print(session["session_id"][:8], session.get("title", ""), session.get("model", ""))

# Search sessions in current profile
for session in dev.sessions(search="code review", limit=5):
    print("match:", session["session_id"][:8], session.get("search_snippet", ""))

# Named profiles are isolated from each other
work = talu.Profile("work")
talu.Chat(MODEL_URI, profile=work).send("Draft a project plan for Q2.")
print(f"dev sessions: {len(dev.sessions(limit=50))}")
print(f"work sessions: {len(work.sessions(limit=50))}")

# Convenience shortcut from top-level namespace
print("Latest dev session:", talu.list_sessions(profile=dev.name, limit=1))
