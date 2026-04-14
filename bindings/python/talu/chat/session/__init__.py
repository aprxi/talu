"""Session management for Chat and AsyncChat."""

from .async_ import AsyncChat
from .sync import Chat, _build_c_storage_records

__all__ = [
    "Chat",
    "AsyncChat",
]

# Internal helper - not part of public API but imported by _chat_base.py
# for ItemRecord conversion.
_build_c_storage_records = _build_c_storage_records
