"""Regression checks for deterministic per-sample eval seed derivation.

Run:
    python bench/test_eval_seed_derivation.py
"""

from __future__ import annotations

from responses.evals._runner import _derive_request_seed

_MAX_I64 = (1 << 63) - 1


def test_same_input_same_seed() -> None:
    sample = {"index": 12, "question_hash": "abcdef0123456789"}
    a = _derive_request_seed(42, sample)
    b = _derive_request_seed(42, sample)
    assert a == b, (a, b)
    assert a != 0, a


def test_changes_with_index() -> None:
    s1 = {"index": 12, "question_hash": "abcdef0123456789"}
    s2 = {"index": 13, "question_hash": "abcdef0123456789"}
    a = _derive_request_seed(42, s1)
    b = _derive_request_seed(42, s2)
    assert a != b, (a, b)


def test_changes_with_question_hash() -> None:
    s1 = {"index": 12, "question_hash": "abcdef0123456789"}
    s2 = {"index": 12, "question_hash": "abcdef0123456790"}
    a = _derive_request_seed(42, s1)
    b = _derive_request_seed(42, s2)
    assert a != b, (a, b)


def test_changes_with_base_seed() -> None:
    sample = {"index": 12, "question_hash": "abcdef0123456789"}
    a = _derive_request_seed(42, sample)
    b = _derive_request_seed(10000, sample)
    assert a != b, (a, b)


def test_endpoint_safe_i64_range() -> None:
    sample = {"index": 999, "question_hash": "ffffffffffffffff"}
    for seed in (1, 42, 1000, 10000, _MAX_I64):
        out = _derive_request_seed(seed, sample)
        assert 1 <= out <= _MAX_I64, out


if __name__ == "__main__":
    tests = [
        test_same_input_same_seed,
        test_changes_with_index,
        test_changes_with_question_hash,
        test_changes_with_base_seed,
        test_endpoint_safe_i64_range,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: OK")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{test.__name__}: FAIL: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
