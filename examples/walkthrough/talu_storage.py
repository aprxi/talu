"""TaluDB chat storage walkthrough (user-facing).

TaluDB is a high-performance *file format*, not a server. A folder is your
chat database. Point Database at a path and it creates or reuses the files
inside that folder.

Persistence is automatic:
- each append is written to the WAL immediately (crash-safe)
- data is flushed into current.talu automatically as buffers fill
- reopening the same folder restores the conversation

CLI Alternative
---------------
For command-line workflows, see ``examples/cli/``:

    # Initialize storage
    talu db init ./my-chat-db

    # Generate with persistence
    talu generate --db ./my-chat-db "Hello TaluDB"

    # List and inspect sessions
    talu db list ./my-chat-db
    talu db show <session-id> ./my-chat-db

The CLI uses the same TaluDB format, so databases are interoperable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talu import Chat
from talu.db import Database


# ---------------------------------------------------------------------------
# 1) Create/Open a persistent store
# ---------------------------------------------------------------------------
# A folder is your database. If it doesn't exist, it is created.
# If it does exist, TaluDB loads the data inside it.
#
# storage = Database("talu://./my-chat-db")
# chat = Chat(storage=storage, session_id="demo")
# ---------------------------------------------------------------------------


def _make_db_location(path: Path) -> str:
    return f"talu://{path}"


# ---------------------------------------------------------------------------
# 2) Add data (chat items)
# ---------------------------------------------------------------------------
# Chat.append(...) persists messages immediately to TaluDB.
# ---------------------------------------------------------------------------


def demo_chat_flow(db_root: Path, session_id: str) -> None:
    storage = Database(_make_db_location(db_root))
    chat = Chat(storage=storage, session_id=session_id)
    try:
        chat.append("user", "Hello TaluDB")
        chat.append("assistant", "Hello! This is persisted on disk.")
        print(f"Items now in session: {len(chat.items)}")
    finally:
        chat.close()


# ---------------------------------------------------------------------------
# 3) Inspect what is stored
# ---------------------------------------------------------------------------
# Reopen the same folder + session_id to restore history.
# ---------------------------------------------------------------------------


def demo_restore(db_root: Path, session_id: str) -> None:
    storage = Database(_make_db_location(db_root))
    chat = Chat(storage=storage, session_id=session_id)
    try:
        print(f"Restored {len(chat.items)} items")
        for item in chat.items:
            print(f"- {item.role}: {item.content}")
    finally:
        chat.close()


# ---------------------------------------------------------------------------
# 4) Full demo
# ---------------------------------------------------------------------------


def demo_flow(db_root: Path, session_id: str) -> None:
    demo_chat_flow(db_root, session_id)
    demo_restore(db_root, session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="TaluDB chat storage walkthrough.")
    parser.add_argument("--db", type=Path, default=Path("./taludb-demo"))
    parser.add_argument("--session", type=str, default="demo-session")
    parser.add_argument("--mode", choices=["chat", "restore", "demo"], default="demo")
    args = parser.parse_args()

    args.db.mkdir(parents=True, exist_ok=True)
    if args.mode == "chat":
        demo_chat_flow(args.db, args.session)
    elif args.mode == "restore":
        demo_restore(args.db, args.session)
    else:
        demo_flow(args.db, args.session)


if __name__ == "__main__":
    main()
