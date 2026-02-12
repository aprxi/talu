"""Hydrate JSON dicts into dataclass, TypedDict, or Pydantic model instances."""

from __future__ import annotations

import dataclasses
from typing import Any, cast, get_args, get_origin, is_typeddict


def _is_pydantic_model(cls: type | Any) -> bool:
    """Check if cls is a Pydantic BaseModel (without importing pydantic)."""
    return callable(getattr(cls, "model_validate", None))


def dict_to_dataclass(cls: type | Any, data: dict | Any) -> Any:
    """Recursively convert a dictionary to a dataclass, TypedDict, or Pydantic model instance.

    Supports:
    - dataclass: Uses dataclasses.fields() to reconstruct
    - TypedDict: Returns dict as-is (TypedDict is just a typed dict)
    - Pydantic BaseModel: Uses model_validate() (no pydantic import needed)
    """
    if not isinstance(data, dict):
        return data

    # Pydantic v2 models have model_validate method
    model_validate = getattr(cls, "model_validate", None)
    if callable(model_validate):
        return model_validate(data)

    # TypedDict: the dict IS the result, no conversion needed
    if is_typeddict(cls):
        return data

    # Dataclass handling
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        target_type = field_types[key]
        origin = get_origin(target_type)

        if origin is list and get_args(target_type):
            item_type = get_args(target_type)[0]
            if dataclasses.is_dataclass(item_type) and isinstance(value, list):
                kwargs[key] = [dict_to_dataclass(item_type, item) for item in value]
                continue

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            kwargs[key] = dict_to_dataclass(target_type, value)
            continue

        kwargs[key] = value

    return cast(type, cls)(**kwargs)
