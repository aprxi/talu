"""Regression checks for explicit model_uri handling in bench config loading.

Run:
    python bench/test_model_uri_precision.py
"""

from __future__ import annotations

from scenario import load_config


def test_explicit_model_uri_disables_default_precision_matrix() -> None:
    config = load_config(
        "responses/perf/pp512",
        None,
        overrides=["model_uri=Qwen/Qwen3.5-4B-TQ4"],
    )
    assert config["model_uri"] == ["Qwen/Qwen3.5-4B-TQ4"], config
    assert config["precision"] == ["original"], config


def test_explicit_multiple_model_uris_are_preserved() -> None:
    config = load_config(
        "responses/perf/pp512",
        None,
        overrides=[
            "model_uri=Qwen/Qwen3.5-4B-TQ4,Qwen/Qwen3.5-4B-NVFP4",
        ],
    )
    assert config["model_uri"] == [
        "Qwen/Qwen3.5-4B-TQ4",
        "Qwen/Qwen3.5-4B-NVFP4",
    ], config
    assert config["precision"] == ["original"], config


def test_explicit_precision_keeps_requested_matrix() -> None:
    config = load_config(
        "responses/perf/pp512",
        None,
        overrides=[
            "model_uri=Qwen/Qwen3.5-4B",
            "precision=TQ8,TQ4",
        ],
    )
    assert config["model_uri"] == ["Qwen/Qwen3.5-4B"], config
    assert config["precision"] == ["TQ8", "TQ4"], config


def test_default_precision_matrix_still_applies_without_model_override() -> None:
    config = load_config("responses/perf/pp512", None)
    assert config["model_uri"] == ["Qwen/Qwen3.5-0.8B"], config
    assert config["precision"] == ["original", "TQ8", "TQ4"], config


if __name__ == "__main__":
    tests = [
        test_explicit_model_uri_disables_default_precision_matrix,
        test_explicit_multiple_model_uris_are_preserved,
        test_explicit_precision_keeps_requested_matrix,
        test_default_precision_matrix_still_applies_without_model_override,
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
