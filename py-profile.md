# Feature: Python Profile API

**Goal:** Give casual Python users the same profile/session experience the CLI
provides, without exposing `Database`, `group_id`, or storage internals.

**Status:** Design complete, ready for implementation.

---

## 1. User-Facing API

### 1.1 Target Experience

```python
import talu

MODEL = "LiquidAI/LFM2-350M"

# Ephemeral chat — same as today, no persistence
chat = talu.Chat(MODEL)

# Enable persistence by passing a profile
dev = talu.Profile()  # "dev" profile (or TALU_PROFILE env var)
chat = talu.Chat(MODEL, profile=dev, system="You are helpful.")
response = chat.send("Hello!")
print(chat.session_id)  # auto-generated, saved to ~/.talu/db/dev/

# Use a named profile
work = talu.Profile("work")
chat = talu.Chat(MODEL, profile=work, system="You are helpful.")
response = chat.send("What are the benefits of code review?")

# Via Client (profile passed to Client, inherited by all chats)
client = talu.Client(MODEL, profile=dev)
chat = client.chat(system="You are helpful.")
response = chat.send("Hello!")

# Browse sessions from a profile
for s in dev.sessions():
    print(s["session_id"][:8], s.get("title", ""), s.get("model", ""))

# Search
results = dev.sessions(search="code review")

# Resume a previous chat by session_id
chat = talu.Chat(MODEL, profile=dev, session_id="a1b2c3d4")
response = chat.send("Follow up question")

# Profiles are isolated
print(len(dev.sessions()))    # only dev sessions
print(len(work.sessions()))   # only work sessions

# Default profile reads TALU_PROFILE env var, falls back to "dev"
# $ TALU_PROFILE=default python my_script.py  ← shares CLI's profile
```

### 1.2 Design Principles

- **Opt-in persistence.** `talu.Chat(MODEL)` stays ephemeral (unchanged).
  Pass `profile=Profile()` to enable persistence. No breaking changes.

- **Profile is a parameter, not a container.** `Chat` and `Client` remain
  the primary interfaces. Profile is passed to them — it doesn't create
  chats or manage models.

- **No registration required.** Profiles are created on first use.
  `talu.Profile("work")` auto-creates the bucket directory if needed.

- **`TALU_PROFILE` env var.** Same semantics as the CLI. The default
  `Profile()` reads `TALU_PROFILE`, falling back to `"dev"`.
  `TALU_PROFILE=default` shares the CLI's default profile.

### 1.3 API Surface — Profile

Profile is a lightweight storage namespace. It does not manage models or chats.

```python
class Profile:
    """A named storage namespace for chat sessions.

    Each profile maps to a storage directory under ~/.talu/db/<name>/.
    No setup required — created automatically on first use.

    Profile resolution:
    1. Explicit name: Profile("work") → ~/.talu/db/work/
    2. TALU_PROFILE env var: Profile() reads os.environ["TALU_PROFILE"]
    3. Fallback: Profile() with no env var → "dev" → ~/.talu/db/dev/

    Args:
        name: Profile name. If None, reads TALU_PROFILE env var,
              falling back to "dev".
    """

    def __init__(self, name: str | None = None): ...

    @property
    def name(self) -> str:
        """Profile name."""

    @property
    def path(self) -> Path:
        """Storage directory path (~/.talu/db/<name>/)."""

    def sessions(
        self,
        *,
        search: str | None = None,
        limit: int = 50,
    ) -> list[SessionRecord]:
        """List chat sessions in this profile.

        Returns standard SessionRecord dicts.

        Args:
            search: Full-text search query (optional).
            limit: Max sessions to return (default 50, newest first).

        Returns:
            List of SessionRecord dicts. Key fields:
            - session_id: str
            - title: str (may be absent)
            - model: str (may be absent)
            - head_item_id: int (latest item index, 0 if empty)
            - created_at_ms: int (unix ms)
            - updated_at_ms: int (unix ms)
            - marker: str ("pinned", "archived", "deleted", or "")
            - search_snippet: str (present when search= is used)

        Example:
            >>> for s in work.sessions():
            ...     print(s["session_id"][:8], s.get("title", ""))
        """

    def delete(self, session_id: str) -> None:
        """Delete a chat session by ID."""
```

### 1.4 API Changes — Chat and Client

`Chat` and `Client` gain a `profile=` parameter. They remain the primary
interfaces for inference.

```python
# Chat gains profile= parameter
class Chat:
    def __init__(
        self,
        model: str | None = None,
        *,
        profile: Profile | None = None,  # None = ephemeral (unchanged)
        session_id: str | None = None,
        system: str | None = None,
        # ... existing params unchanged ...
    ): ...

# Client gains profile= parameter, passes it to all chats
class Client:
    def __init__(
        self,
        model: ModelInput | list[ModelInput],
        *,
        profile: Profile | None = None,  # None = ephemeral (unchanged)
        # ... existing params unchanged ...
    ): ...

    def chat(
        self,
        *,
        session_id: str | None = None,
        # ... existing params unchanged ...
        # NOTE: profile inherited from Client, not passed per-chat
    ) -> Chat: ...
```

**Default behavior:**

```python
# In Chat.__init__:
if profile is None:
    pass                      # ephemeral, in-memory — same as today
# else: use the Profile instance for persistence
```

- `Chat(MODEL)` → ephemeral, no persistence (unchanged from today)
- `Chat(MODEL, profile=Profile())` → persistent to "dev" profile
- `Chat(MODEL, profile=Profile("work"))` → persistent to "work" profile

When a profile is active, Chat internally:
1. Gets the `Database` from the profile
2. Auto-generates `session_id` if none provided
3. Passes `storage=` and `session_id=` to the existing plumbing

### 1.5 Top-Level Convenience

```python
# Exported from talu namespace
talu.Profile          # the class
talu.list_sessions    # shortcut
```

```python
def list_sessions(
    *,
    profile: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> list[SessionRecord]:
    """List sessions in a profile. Shortcut for Profile(name).sessions()."""
    return Profile(profile).sessions(search=search, limit=limit)
```

### 1.6 Profile Resolution

```
Profile("work")                        → "work"     → ~/.talu/db/work/
Profile("default")                     → "default"  → ~/.talu/db/default/  (CLI's profile)
Profile()  [no env var]                → "dev"      → ~/.talu/db/dev/
Profile()  [TALU_PROFILE=work]         → "work"     → ~/.talu/db/work/
Profile()  [TALU_PROFILE=default]      → "default"  → ~/.talu/db/default/
```

- Explicit `name` argument always wins over the env var.
- `TALU_PROFILE` is checked only when `name` is `None`.
- The fallback `"dev"` avoids colliding with CLI's `"default"`.
- Named profiles like `"work"` are fully shared between CLI and Python.

### 1.7 Backward Compatibility

**No breaking changes.** `talu.Chat(MODEL)` remains ephemeral — identical
to today's behavior. Profile is a new optional parameter (`profile=None`
by default). Existing code is unaffected.

---

## 2. How the CLI Does It (Reference)

The CLI is written in Rust (`bindings/rust/cli/`). Its profile system lives
entirely in the Rust layer — Zig/core has no knowledge of profiles.

### 2.1 Profile Resolution

**File:** `bindings/rust/cli/src/config.rs`

```
TALU_PROFILE=work  →  resolve_profile("work")  →  ~/.talu/db/work/
```

- Config lives at `~/.talu/config.toml`
- Each profile maps to a bucket directory
- Default bucket path: `~/.talu/db/<name>/`
- `"default"` profile is auto-created if absent

### 2.2 Bucket Initialization

**Function:** `ensure_bucket()` in `config.rs`

When a profile's directory doesn't exist, the CLI creates:

```
~/.talu/db/<name>/
├── store.key         # 16 random bytes (getrandom)
└── manifest.json     # {"version": 1, "segments": [], "last_compaction_ts": 0}
```

This initialization is the **caller's responsibility** — Zig expects the bucket
to exist and be initialized before `set_storage_db` is called.

### 2.3 CLI Ask Flow

**File:** `bindings/rust/cli/src/cli/ask.rs`

```
1. resolve_bucket(no_bucket, bucket_override, profile)  → bucket_path
2. ChatHandle::new(system)                               → chat handle
3. chat.set_storage_db(bucket_path, session_id)          → attach storage
4. chat.notify_session_update(model, title, marker)      → persist metadata
5. chat.generate(...)                                    → inference
```

- `set_storage_db` attaches TaluDB storage AND loads existing items if the
  session already exists (resume).
- `notify_session_update` persists session metadata (model name, title, marker).
- The CLI generates session IDs via `talu::responses::new_session_id()`.

### 2.4 Session Listing

**File:** `bindings/rust/cli/src/cli/sessions.rs`

`talu ask` with no args prints a table:

```
SESSION ID                           | TITLE            | MODEL          | ITEMS   | TURNS   | TOKENS   | UPDATED
------------------------------------------------------------------------------------------------------------------------
a1b2c3d4-e5f6-...                    | What are the b.. | LFM2-350M      | 4       | 2       | 512      | 5 min ago
```

- `session_id`, `title`, `model`, `updated_at` come from `SessionRecord` metadata
- `items`, `turns`, `tokens` are computed by loading each conversation via
  `StorageHandle::load_conversation()` and iterating items

In Python, `profile.sessions()` returns the `SessionRecord` dicts. The ITEMS/
TURNS/TOKENS stats would require a separate conversation-load step (future work).

---

## 3. What Zig/Core Knows

Zig has **zero knowledge of profiles**. It provides a storage engine with:

### 3.1 C API Functions Used

| C Function | Purpose |
|------------|---------|
| `talu_chat_set_storage_db(chat, path, session_id)` | Attach TaluDB storage to a chat. If session exists, loads all items (resume). If new, starts empty. |
| `talu_chat_notify_session_update(chat, model, title, system, config_json, marker, parent_session_id, group_id, metadata_json, source_doc_id)` | Persist/update session metadata. |
| `talu_storage_list_sessions_ex(path, limit, ..., search_query, ..., group_id, ...)` | List sessions with filtering. |
| `talu_storage_free_sessions(sessions)` | Free session list memory. |

### 3.2 Session Resume Already Works

In `bindings/python/talu/chat/_chat_base.py`, lines 181-192:

```python
# Attach TaluDB storage — Zig loads existing items if session exists
result = self._lib.talu_chat_set_storage_db(
    self._chat_ptr,
    db_path.encode("utf-8"),
    self._session_id.encode("utf-8"),
)
check(result, {"db_path": db_path, "session_id": self._session_id})

# Detect new vs resumed session
if len(self.items) == 0:
    # New session: set system prompt
    if system is not None:
        self.system = system
else:
    # Resumed session: read system prompt from loaded items
    self._system = self.items.system
```

No core changes needed. The resume path is already functional.

### 3.3 No Core Changes Required

Confirmed by tracing the full path:

- `talu_chat_set_storage_db` — exists, handles attach + resume
- `talu_storage_list_sessions_ex` — exists, handles list/search/filter
- Session ID generation — Python can use `uuid.uuid4()` or expose
  `talu_responses_new_session_id` from Zig
- Bucket initialization — caller responsibility (CLI does it in Rust,
  Python will do it in Python)

---

## 4. Existing Python Plumbing

### 4.1 What Already Works

| Component | Location | Status |
|-----------|----------|--------|
| `Chat(session_id=, storage=Database("talu://..."))` | `chat/session/sync.py` | Works — creates or resumes a session |
| `Database("talu://path")` | `db/backends.py` | Works — wraps TaluDB path |
| `Database.list_sessions(search_query=, limit=)` | `db/backends.py` | Works — full filtering support |
| `Chat.session_id` property | `chat/_chat_base.py` | Works — readable after creation |
| Session resume detection | `chat/_chat_base.py:188-192` | Works — checks `len(items)` after storage attach |

### 4.2 What Needs to Be Built

| Component | Description |
|-----------|-------------|
| `Profile` class | New class: name → bucket path → Database, with `sessions()`, `delete()` |
| `profile=` on `Chat` | New parameter on `Chat.__init__` (default: `Profile()`) |
| `profile=` on `Client` | New parameter on `Client.__init__`, inherited by `client.chat()` |
| Bucket initialization | Python equivalent of CLI's `ensure_bucket()`: create dir + `store.key` + `manifest.json` |
| Session ID auto-generation | When profile is active and no `session_id` given, auto-generate UUID |
| `talu.list_sessions()` | Top-level convenience function |
| Export from `talu/__init__.py` | Add `Profile` and `list_sessions` to `__all__` |

### 4.3 Known Issues to Fix

**`group_id` not persisted in `notify_session_update`:**

In `_chat_base.py` line 214, `group_id` is hardcoded to `None` even though
`self._group_id` is stored. Not blocking for the Profile feature (profiles
use directory isolation, not `group_id`), but should be fixed separately.

**`Client.chat()` doesn't forward `storage` or `group_id`:**

This will be addressed as part of this feature — `Client` gains `profile=`
and forwards the derived `storage` to all chats it creates.

---

## 5. Implementation Plan

### 5.1 New File: `bindings/python/talu/profile.py`

```python
"""Chat profiles — named storage namespaces for conversations."""

import json
import os
import uuid
from pathlib import Path

from .db import Database

# Same base as CLI: ~/.talu/db/<name>/  (hardcoded, no TALU_HOME override)
_DB_BASE = Path.home() / ".talu" / "db"

# Python's default profile name (intentionally not "default" to avoid
# colliding with the CLI's default profile).
_PYTHON_DEFAULT_PROFILE = "dev"


def _resolve_profile_name(name: str | None) -> str:
    """Resolve profile name: explicit arg > TALU_PROFILE env > "dev"."""
    if name is not None:
        return name
    return os.environ.get("TALU_PROFILE", _PYTHON_DEFAULT_PROFILE)


def _ensure_bucket(path: Path) -> None:
    """Initialize a storage bucket directory if it doesn't exist.

    Creates the directory with store.key and manifest.json,
    matching the CLI's ensure_bucket() behavior in config.rs.
    """
    if path.exists():
        return

    path.mkdir(parents=True, exist_ok=True)

    # store.key: 16 random bytes
    key_path = path / "store.key"
    key_path.write_bytes(os.urandom(16))

    # manifest.json: empty initial manifest
    manifest_path = path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "segments": [], "last_compaction_ts": 0})
    )


def new_session_id() -> str:
    """Generate a new session ID (UUID4 string)."""
    return str(uuid.uuid4())


class Profile:
    """A named storage namespace for chat sessions.

    ...docstring per section 1.3...
    """

    def __init__(self, name: str | None = None):
        self._name = _resolve_profile_name(name)
        self._path = _DB_BASE / self._name
        _ensure_bucket(self._path)
        self._db = Database(f"talu://{self._path}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def database(self) -> Database:
        """The underlying Database (for internal use by Chat/Client)."""
        return self._db

    def sessions(self, *, search=None, limit=50):
        return self._db.list_sessions(search_query=search, limit=limit)

    def delete(self, session_id):
        # Calls talu_storage_delete_session (already bound in _native.py)
        ...

    def __repr__(self):
        return f"Profile({self._name!r})"
```

### 5.2 Changes to `Chat.__init__` in `chat/session/sync.py`

```python
from ..profile import Profile, new_session_id

class Chat(ChatBase):
    def __init__(
        self,
        model: str | None = None,
        *,
        profile: Profile | None = None,
        # ... existing params ...
    ):
        # If profile is active, wire up storage and session_id
        if profile is not None:
            if storage is not None:
                raise ValidationError("Cannot use both 'profile' and 'storage'")
            storage = profile.database
            if session_id is None:
                session_id = new_session_id()

        # ... rest of existing __init__ unchanged ...
```

### 5.3 Changes to `Client` in `client.py`

```python
class Client:
    def __init__(
        self,
        model: ModelInput | list[ModelInput],
        *,
        profile: Profile | None = None,
        # ... existing params ...
    ):
        self._profile = profile
        # ... rest unchanged ...

    def chat(self, *, session_id=None, **kwargs) -> Chat:
        # ... existing logic ...
        chat_instance = Chat(
            client=self,
            profile=self._profile,  # pass through
            session_id=session_id,
            **kwargs,
        )
        # ...
```

### 5.4 Export from `talu/__init__.py`

```python
from talu.profile import Profile

def list_sessions(*, profile=None, search=None, limit=50):
    return Profile(profile).sessions(search=search, limit=limit)

__all__ = [
    ...
    # Profile
    "Profile",
    "list_sessions",
    ...
]
```

### 5.5 Session Title Auto-Generation

The CLI auto-generates titles from the first ~50 characters of the first prompt
(see `ask.rs` lines 554-558). Chat should do the same when a profile is active.

Options:
- Extend `_chat_base.py` to call `notify_session_update` with a derived title
  after the first `send()` call.
- Use the existing Hook system to intercept first generation.

### 5.6 Session Delete

`talu_storage_delete_session(db_path, session_id)` exists in the C API and
is already bound in `_native.py`. `Profile.delete()` calls it directly.

---

## 6. Example: `examples/python/10_manage_profiles.py`

**File:** `examples/python/10_manage_profiles.py`

```python
"""Profiles and sessions — save, list, and resume chats.

This example shows:
- How the default "dev" profile works (persistence out of the box)
- Browsing sessions (list, search, filter)
- Resuming a previous conversation
- Working with session metadata
- Using named profiles for isolation

Profiles are created automatically — no setup needed.
"""

import os
import sys
from datetime import datetime, timezone

import talu
from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

if not repository.is_cached(MODEL_URI):
    sys.exit(f"Model '{MODEL_URI}' not found. Run: python examples/python/00_fetch_model.py")


# --- Enable persistence with the default profile ---

dev = talu.Profile()  # "dev" profile — or set TALU_PROFILE env var
chat = talu.Chat(MODEL_URI, profile=dev, system="You are a helpful assistant.")
response = chat.send("What are the benefits of code review?")
print(response)
print(f"Session ID: {chat.session_id}")

# Follow-up in the same session
response = response.append("Summarize in 3 bullet points.")
print(response)


# --- Browse sessions ---

# sessions() returns a list of SessionRecord dicts
sessions = dev.sessions()
print(f"\n{len(sessions)} session(s) in '{dev.name}' profile:\n")

for s in sessions:
    sid = s["session_id"][:8]
    title = s.get("title", "(untitled)")
    model = s.get("model", "")
    updated = s.get("updated_at_ms", 0)

    # Timestamps are unix milliseconds — convert as needed
    if updated:
        dt = datetime.fromtimestamp(updated / 1000, tz=timezone.utc)
        when = dt.strftime("%Y-%m-%d %H:%M")
    else:
        when = "-"

    print(f"  {sid}  {title:<40}  {model:<16}  {when}")


# --- Search sessions ---

results = dev.sessions(search="code review")
print(f"\nSearch 'code review': {len(results)} match(es)")
for s in results:
    # search_snippet is populated when search= is used
    snippet = s.get("search_snippet", "")
    print(f"  {s['session_id'][:8]}  {snippet}")


# --- Resume a previous session ---

saved_id = chat.session_id
chat2 = talu.Chat(MODEL_URI, profile=dev, session_id=saved_id)

# chat2 has the full conversation history from above
print(f"\nResumed session {saved_id[:8]} with {len(chat2.items)} items")
response = chat2.send("Make it 5 bullet points instead.")
print(response)


# --- Via Client (profile inherited by all chats) ---

client = talu.Client(MODEL_URI, profile=dev)
chat3 = client.chat(system="You are concise.")
response = chat3.send("What is Python?")
print(f"\nClient chat session: {chat3.session_id[:8]}")
print(response)


# --- Find the most recent session ---

latest = dev.sessions(limit=1)
if latest:
    print(f"\nMost recent: {latest[0].get('title', '')} ({latest[0]['session_id'][:8]})")


# --- Named profiles for isolation ---

work = talu.Profile("work")
talu.Chat(MODEL_URI, profile=work).send("Draft a project plan.")

dev_count = len(dev.sessions())
work_count = len(work.sessions())
print(f"\nDev: {dev_count} session(s)")
print(f"Work: {work_count} session(s)")
# Each profile is isolated — dev doesn't see work sessions


# --- Opt out of persistence ---

# profile=None makes the chat ephemeral (in-memory only, nothing saved)
chat_ephemeral = talu.Chat(MODEL_URI, profile=None)
chat_ephemeral.send("This won't be saved anywhere.")
```

---

## 7. Storage Layout

```
~/.talu/
├── config.toml              # CLI config (profiles, default_model)
├── db/
│   ├── default/             # CLI default profile
│   │   ├── store.key
│   │   ├── manifest.json
│   │   └── *.seg            # TaluDB segment files
│   ├── dev/                 # Python default profile
│   │   ├── store.key
│   │   ├── manifest.json
│   │   └── ...
│   ├── work/                # Named profile (shared by CLI + Python)
│   │   └── ...
│   └── personal/            # Named profile
│       └── ...
```

Named profiles (like "work") are shared between CLI and Python — they use
the same bucket format and the same Zig storage engine. A session created
via Python is visible via `talu ask --profile work` in the CLI, and vice versa.

Only the **unnamed defaults** are separate ("default" for CLI, "dev" for Python).

---

## 8. Testing

### 8.1 Unit Tests

| Test | What it verifies |
|------|-----------------|
| `Profile("test")` creates bucket | Directory + `store.key` + `manifest.json` exist |
| `Profile("test")` is idempotent | Second call doesn't overwrite `store.key` |
| `Chat(model, profile=p)` has `session_id` set | Auto-generated UUID |
| `Chat(model, profile=p, session_id=X)` | Uses given ID |
| `profile.sessions()` returns list | Empty for new profile |
| `profile.sessions(search=)` filters | Only matching results |
| `Profile()` defaults to `"dev"` | Name and path match |
| `TALU_PROFILE` env var override | Profile() reads it |
| `Chat(model)` without profile | Ephemeral, no persistence (unchanged) |
| `profile=` and `storage=` mutually exclusive | Raises ValidationError |

### 8.2 Integration Tests

| Test | What it verifies |
|------|-----------------|
| Chat → send → profile.sessions() shows session | End-to-end persistence |
| Chat → send → new Chat with same session_id → items has history | Session resume |
| Two profiles → sessions are isolated | No cross-contamination |
| Client(profile=) → client.chat() → persists | Profile inherited by Client chats |
| Resume with system prompt → system preserved | `items.system` matches |

### 8.3 Test Location

`bindings/python/tests/profile/test_profile.py`

Use `tmp_path` fixture to override `TALU_HOME` so tests don't pollute
`~/.talu/`.

---

## 9. Resolved Decisions

1. **Profile is a parameter, not a container.** Pass `profile=` to Chat/Client.
   Profile doesn't create chats — Chat and Client remain the primary interfaces.

2. **Opt-in persistence.** `talu.Chat(MODEL)` stays ephemeral (no breaking
   change). Pass `profile=Profile()` to enable persistence.

3. **No registration required.** Profiles auto-create on first use.

4. **`TALU_PROFILE` env var.** Same semantics as the CLI. `Profile()` reads
   it, falls back to `"dev"`. Explicit name argument always wins.

5. **No core/Zig changes.** Everything is Python-side plumbing over existing
   C API functions.

## 10. Resolved Questions

1. **Session delete:** `talu_storage_delete_session` exists in the C API
   and is already bound in `_native.py`. `Profile.delete()` calls it directly.

2. **Title generation:** Out of scope for this feature. Title can be set
   via `notify_session_update` — follow-up work if needed.

3. **Session ID format:** Use Python `uuid.uuid4()`. Zig's
   `generateSessionId` is standard UUIDv4 (RFC 4122) — same format.
   `talu_session_id_new` is also bound in `_native.py` if we want to
   use it, but `uuid.uuid4()` produces identical output.

4. **`TALU_HOME` for profiles:** No. The CLI hardcodes `~/.talu/db/<name>/`
   (see `config.rs:70`). Python should match. `TALU_HOME` is only used
   for model cache paths, not database/profile paths.

5. **Backward compatibility:** Profile is opt-in (`profile=None` by default).
   No breaking changes. Existing code unaffected.
