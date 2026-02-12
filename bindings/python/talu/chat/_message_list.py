"""MessageList - Immutable view of messages with pretty REPL display."""

from __future__ import annotations


class MessageList(list):
    """
    An immutable view of message dicts with pretty __repr__ for REPL/notebook use.

    This provides a familiar list interface while preventing accidental
    mutation of the conversation history. Modifications must go through
    Chat methods (clear, reset, append, etc.).

    The pretty display shows messages in a readable format:
        [
          system: You are helpful.
          user: Hello!
          assistant: Hi there!
        ]

    But it's still a list, so standard access works:
        >>> messages[0]
        {'role': 'system', 'content': 'You are helpful.'}
        >>> len(messages)
        3
    """

    def __repr__(self) -> str:
        if not self:
            return "[]"

        lines = ["["]
        for msg in self:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Escape newlines for display
            content = content.replace("\n", "\\n")
            # Truncate long content for display
            if len(content) > 80:
                content = content[:77] + "..."
            lines.append(f"  {role}: {content}")
        lines.append("]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # Block mutation methods with clear error messages
    def append(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError(
            "MessageList is read-only. Use Chat methods to modify: "
            "add_user_message(), add_assistant_message(), pop(), clear()."
        )

    def extend(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def insert(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only. Use Chat.insert() instead.")

    def remove(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only. Use Chat.remove() instead.")

    def pop(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only. Use Chat.pop() instead.")

    def clear(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only. Use Chat.clear() instead.")

    def __setitem__(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def __delitem__(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def __iadd__(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def __imul__(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def sort(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")

    def reverse(self, *_args: object, **_kwargs: object) -> None:  # type: ignore[override]
        raise TypeError("MessageList is read-only.")
