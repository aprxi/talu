"""Regression checks for penalty forwarding in eval request builders.

Run:
    python bench/test_eval_penalty_forwarding.py
"""

from __future__ import annotations

from responses.evals import bfcl, gpqa, ifeval, mmlu, mmmu


def _assert_penalties_present(body: dict) -> None:
    assert body.get("presence_penalty") == 1.5, body
    assert body.get("frequency_penalty") == 0.3, body


def test_mmlu_forwards_penalties() -> None:
    body = mmlu._build_body(
        {"prompt": "Q?\nA. a\nB. b\nC. c\nD. d"},
        "Qwen/Qwen3.5-2B",
        {"presence_penalty": 1.5, "frequency_penalty": 0.3},
    )
    _assert_penalties_present(body)


def test_gpqa_forwards_penalties() -> None:
    body = gpqa._build_body(
        {"prompt": "Q?\nA. a\nB. b\nC. c\nD. d"},
        "Qwen/Qwen3.5-2B",
        {"presence_penalty": 1.5, "frequency_penalty": 0.3},
    )
    _assert_penalties_present(body)


def test_ifeval_forwards_penalties() -> None:
    body = ifeval._build_body(
        {"prompt": "Do X."},
        "Qwen/Qwen3.5-2B",
        {"presence_penalty": 1.5, "frequency_penalty": 0.3},
    )
    _assert_penalties_present(body)


def test_bfcl_forwards_penalties() -> None:
    body = bfcl._build_body(
        {"prompt": "Call tool", "tools": [], "scoring_mode": "simple"},
        "Qwen/Qwen3.5-2B",
        {"presence_penalty": 1.5, "frequency_penalty": 0.3},
    )
    _assert_penalties_present(body)


def test_mmmu_forwards_penalties() -> None:
    build_body = mmmu._make_build_body("http://127.0.0.1:18258")
    body = build_body(
        {"prompt": "Q?\nA. a\nB. b\nC. c\nD. d", "images": []},
        "Qwen/Qwen3.5-2B",
        {"presence_penalty": 1.5, "frequency_penalty": 0.3},
    )
    _assert_penalties_present(body)


if __name__ == "__main__":
    tests = [
        test_mmlu_forwards_penalties,
        test_gpqa_forwards_penalties,
        test_ifeval_forwards_penalties,
        test_bfcl_forwards_penalties,
        test_mmmu_forwards_penalties,
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
