"""Regression checks for eval session-based log paths.

Run:
    python bench/test_eval_session.py
"""

from __future__ import annotations

from pathlib import Path

from log import eval_log_path


def test_new_run_has_no_session_suffix() -> None:
    p = eval_log_path("mmlu", "Qwen/Qwen3-0.6B", 100, 0)
    assert "_s" not in p.stem, p


def test_session_suffix_is_added() -> None:
    p = eval_log_path("mmlu", "Qwen/Qwen3-0.6B", 100, 0, session_id="abc123")
    assert p.stem.endswith("_sabc123"), p


def test_session_suffix_is_sanitized() -> None:
    p = eval_log_path("mmlu", "Qwen/Qwen3-0.6B", 100, 0, session_id="../a b")
    assert p.stem.endswith("_s___a_b"), p
    assert str(p).startswith(str(Path(__file__).resolve().parent.parent / "temp" / "evals"))


if __name__ == "__main__":
    tests = [
        test_new_run_has_no_session_suffix,
        test_session_suffix_is_added,
        test_session_suffix_is_sanitized,
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
