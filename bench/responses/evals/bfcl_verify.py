"""AST-based verification for BFCL function-calling evaluation.

Compares model-generated tool calls against BFCL ground truth using
structural matching with type coercion and value normalization.

Ground truth format (per sample):
    [{"func_name": {"param": [acceptable_value, ...], ...}}, ...]

Each parameter maps to a list of acceptable values.  An empty string ""
in the list means the parameter may be omitted.
"""

from __future__ import annotations

import json
import re
import math


def _normalize_str(s: str) -> str:
    """Lowercase, strip whitespace and common punctuation for fuzzy string match."""
    s = s.strip().lower()
    # Remove surrounding quotes.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s)
    return s


def _param_matches(actual, acceptable: list) -> bool:
    """Check if *actual* matches any entry in *acceptable*.

    Type coercion rules:
        - int ↔ float (5 == 5.0)
        - bool compared strictly (True ≠ 1)
        - strings compared after normalization
        - dicts compared recursively (keys + values)
        - lists compared element-wise (order matters)
    """
    for expected in acceptable:
        if _values_equal(actual, expected):
            return True
    return False


def _values_equal(actual, expected) -> bool:
    """Deep equality with type coercion."""
    # Bool must match exactly (before == check since bool is int subclass in Python).
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected

    # Identical.
    if actual == expected:
        return True

    # Empty string sentinel means "may be omitted" — handled by caller.
    if expected == "":
        return True

    # Numeric: int ↔ float.
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        # Handle NaN.
        if math.isnan(float(actual)) and math.isnan(float(expected)):
            return True
        return float(actual) == float(expected)

    # String normalization.
    if isinstance(actual, str) and isinstance(expected, str):
        return _normalize_str(actual) == _normalize_str(expected)

    # String ↔ number: try parsing the string.
    if isinstance(actual, str) and isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (ValueError, TypeError):
            return False
    if isinstance(actual, (int, float)) and isinstance(expected, str):
        try:
            return float(actual) == float(expected)
        except (ValueError, TypeError):
            return False

    # Dict: recursive key-value match.
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_values_equal(actual[k], expected[k]) for k in actual)

    # List: element-wise.
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(_values_equal(a, e) for a, e in zip(actual, expected))

    return False


def _call_matches(actual_call: dict, gt_entry: dict, func_params: dict | None = None) -> bool:
    """Check if a single actual call matches a ground truth entry.

    Args:
        actual_call: {"name": str, "arguments": dict}
        gt_entry:    {"func_name": {"param": [acceptable_values], ...}}
        func_params: Optional parameter schema for determining required vs optional.
    """
    if len(gt_entry) != 1:
        return False

    gt_name = next(iter(gt_entry))
    gt_params = gt_entry[gt_name]

    # Name match (allow underscore ↔ dot variation).
    actual_name = actual_call.get("name", "")
    if actual_name != gt_name:
        # Try underscore → dot.
        if actual_name.replace("_", ".") != gt_name and actual_name != gt_name.replace(".", "_"):
            return False

    actual_args = actual_call.get("arguments", {})
    if isinstance(actual_args, str):
        try:
            actual_args = json.loads(actual_args)
        except (json.JSONDecodeError, TypeError):
            return False

    # Determine required parameters.
    required = set()
    if func_params and "required" in func_params:
        required = set(func_params["required"])

    # Check every ground truth parameter.
    for param_name, acceptable_values in gt_params.items():
        if param_name in actual_args:
            if not _param_matches(actual_args[param_name], acceptable_values):
                return False
        else:
            # Missing parameter: acceptable only if "" is in the acceptable list
            # (meaning the param is optional and can be omitted).
            if "" not in acceptable_values:
                return False

    return True


def match_simple(actual_calls: list[dict], ground_truth: list[dict],
                 func_schemas: list[dict] | None = None) -> bool:
    """Verify a single function call matches ground truth (simple/multiple categories).

    Args:
        actual_calls: List of parsed function calls from model output.
        ground_truth: List with exactly 1 entry: [{"func_name": {"param": [values]}}].
        func_schemas: Optional list of function schemas (for required/optional info).
    """
    if len(actual_calls) != 1 or len(ground_truth) != 1:
        return False

    func_params = None
    if func_schemas:
        gt_name = next(iter(ground_truth[0]))
        for schema in func_schemas:
            if schema.get("name") == gt_name:
                func_params = schema.get("parameters", {})
                break

    return _call_matches(actual_calls[0], ground_truth[0], func_params)


def match_parallel(actual_calls: list[dict], ground_truth: list[dict],
                   func_schemas: list[dict] | None = None) -> bool:
    """Verify multiple function calls match ground truth (order-independent).

    Each ground truth entry must be matched by exactly one actual call.
    """
    if len(actual_calls) != len(ground_truth):
        return False

    # Greedy bipartite matching: for each GT entry, find a matching actual call.
    used = [False] * len(actual_calls)
    for gt_entry in ground_truth:
        gt_name = next(iter(gt_entry))
        func_params = None
        if func_schemas:
            for schema in func_schemas:
                if schema.get("name") == gt_name:
                    func_params = schema.get("parameters", {})
                    break

        matched = False
        for j, ac in enumerate(actual_calls):
            if used[j]:
                continue
            if _call_matches(ac, gt_entry, func_params):
                used[j] = True
                matched = True
                break
        if not matched:
            return False

    return True


def match_irrelevance(actual_calls: list[dict]) -> bool:
    """Irrelevance detection: model must NOT produce any function calls."""
    return len(actual_calls) == 0
