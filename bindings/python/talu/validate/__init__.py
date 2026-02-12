"""
JSON schema validation with streaming support.

This module provides schema-based JSON validation, with a unique capability:
**streaming validation** - validate JSON byte-by-byte as it arrives.

This is particularly useful when consuming streaming responses from LLM APIs,
where you want to detect schema violations early and abort costly requests.

Quick Start
-----------

Validate complete JSON::

    from pydantic import BaseModel
    from talu.validate import Validator

    class User(BaseModel):
        name: str
        age: int

    validator = Validator(User)
    validator.validate('{"name":"Alice","age":30}')  # True
    validator.validate('{"name":123}')  # False (wrong type)

Streaming validation::

    validator = Validator(User)

    # Feed chunks as they arrive from an API
    validator.feed('{"name":')
    validator.feed('"Alice",')
    validator.feed('"age":30}')

    validator.is_complete  # True - valid complete JSON

Early abort on schema violation::

    validator = Validator(User)
    validator.feed('{"name":')
    validator.feed('123')  # Returns False - number not allowed here
    # Stop streaming, save API costs!

See Also
--------
talu.Chat : Use response_format for automatic validation during generation.
"""

from .validator import Validator

__all__ = ["Validator"]
