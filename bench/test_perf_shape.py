"""Regression checks for perf scenario naming and auto-shape inference.

Run:
    python bench/test_perf_shape.py
"""

from __future__ import annotations

import responses  # noqa: F401

from run import _infer_perf_shape
from scenario import scenario_names


def test_existing_perf_shape_inference_still_works() -> None:
    assert _infer_perf_shape("responses/perf/pp4096b8") == (4096, 8)
    assert _infer_perf_shape("responses/perf/tg512b4") == (512, 4)


def test_mixed_perf_shape_inference_uses_prompt_plus_output_budget() -> None:
    assert _infer_perf_shape("responses/perf/pptg4096x512") == (4608, 1)
    assert _infer_perf_shape("responses/perf/pptg4096x512b4") == (4608, 4)


def test_mixed_perf_scenarios_are_registered() -> None:
    names = set(scenario_names())
    assert "responses/perf/pptg512x128" in names
    assert "responses/perf/pptg2048x512" in names
    assert "responses/perf/pptg4096x512b8" in names


if __name__ == "__main__":
    tests = [
        test_existing_perf_shape_inference_still_works,
        test_mixed_perf_shape_inference_uses_prompt_plus_output_budget,
        test_mixed_perf_scenarios_are_registered,
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
