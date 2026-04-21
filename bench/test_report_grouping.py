"""Regression checks for perf report grouping behavior.

Run:
    python bench/test_report_grouping.py
"""

from __future__ import annotations

from rich.console import Console

from run import _print_combined_model_table, _uses_combined_model_table


def test_single_model_uses_separate_table() -> None:
    rows = [
        {"model": "Qwen/Qwen3.5-4B", "model_uri": "Qwen/Qwen3.5-4B-TQ4"},
        {"model": "Qwen/Qwen3.5-4B", "model_uri": "Qwen/Qwen3.5-4B-TQ4"},
    ]
    assert _uses_combined_model_table(rows) is False


def test_multiple_exact_model_uris_use_combined_table() -> None:
    rows = [
        {"model": "Qwen/Qwen3.5-4B-TQ4", "model_uri": "Qwen/Qwen3.5-4B-TQ4"},
        {"model": "Qwen/Qwen3.5-4B-NVFP4", "model_uri": "Qwen/Qwen3.5-4B-NVFP4"},
    ]
    assert _uses_combined_model_table(rows) is True


def test_combined_table_renders_scenario_title_and_model_column() -> None:
    rows = [
        {
            "model": "Qwen/Qwen3.5-4B-TQ4",
            "model_uri": "Qwen/Qwen3.5-4B-TQ4",
            "engine_tok_s": 199.8,
            "prefill_tok_s": 6423.0,
            "ttft_ms": 671.0,
            "input_tokens": 4094,
            "output_tokens": 3,
            "model_info": {"file_size_bytes": int(3.6 * (1 << 30))},
        },
        {
            "model": "Qwen/Qwen3.5-4B-NVFP4",
            "model_uri": "Qwen/Qwen3.5-4B-NVFP4",
            "engine_tok_s": 169.2,
            "prefill_tok_s": 6283.7,
            "ttft_ms": 685.0,
            "input_tokens": 4094,
            "output_tokens": 3,
            "model_info": {"file_size_bytes": int(3.2 * (1 << 30))},
        },
    ]
    console = Console(record=True, width=200)

    _print_combined_model_table(console, "responses/perf/pp4096", rows)

    rendered = console.export_text()
    assert "responses/perf/pp4096" in rendered
    assert "Model" in rendered
    assert "Precision" not in rendered
    assert rendered.index("Qwen/Qwen3.5-4B-TQ4") < rendered.index("Qwen/Qwen3.5-4B-NVFP4")


if __name__ == "__main__":
    tests = [
        test_single_model_uses_separate_table,
        test_multiple_exact_model_uris_use_combined_table,
        test_combined_table_renders_scenario_title_and_model_column,
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
