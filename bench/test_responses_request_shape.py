"""Regression checks for talu bench `/v1/responses` request bodies.

Run:
    python bench/test_responses_request_shape.py
"""

from __future__ import annotations

from responses.evals import gpqa, ifeval, mmlu, mmmu
from responses.evals._api import format_request
from responses.perf import generate, prefill


def test_eval_api_adapter_sets_store_false() -> None:
    path, body = format_request(
        {
            "model": "Qwen/Qwen3.5-2B",
            "input": "hello",
            "system": "be concise",
            "max_completion_tokens": 4,
            "max_reasoning_tokens": 0,
        }
    )
    assert path == "/v1/responses"
    assert body.get("store") is False, body


def test_eval_builders_set_store_false() -> None:
    sample = {"prompt": "Q?\nA. a\nB. b\nC. c\nD. d", "images": []}
    uri = "Qwen/Qwen3.5-2B"
    config = {"max_reasoning_tokens": 0}

    assert mmlu._build_body(sample, uri, config).get("store") is None
    assert gpqa._build_body(sample, uri, config).get("store") is False
    assert ifeval._build_body({"prompt": "Do X."}, uri, config).get("store") is False

    build_body = mmmu._make_build_body("http://127.0.0.1:18258")
    assert build_body(sample, uri, config).get("store") is False


def _capture_perf_body(module, run_factory, *factory_args) -> dict:
    captured: dict = {}
    original_http_post_stream = module.http_post_stream
    original_extract_generation_metrics = module.extract_generation_metrics
    try:
        def fake_http_post_stream(url: str, body: dict):
            captured.update(body)
            return ([], 0.0)

        def fake_extract_generation_metrics(events: list[dict]) -> dict:
            return {
                "engine_tok_s": 0.0,
                "decode_s": 0.0,
                "output_tokens": 0,
                "input_tokens": 0,
                "prefill_tok_s": 0.0,
                "prefill_ms": 0.0,
                "ttft_ms": 0.0,
                "model_info": {},
            }

        module.http_post_stream = fake_http_post_stream
        module.extract_generation_metrics = fake_extract_generation_metrics

        run = run_factory(*factory_args)
        run(
            None,
            "http://127.0.0.1:18258",
            1,
            {
                "model_uri": ["Qwen/Qwen3.5-2B"],
                "precision": ["original"],
                "streaming": False,
            },
        )
    finally:
        module.http_post_stream = original_http_post_stream
        module.extract_generation_metrics = original_extract_generation_metrics
    return captured


def test_perf_builders_set_store_false() -> None:
    generate_body = _capture_perf_body(generate, generate._make_run, 8, 1)
    prefill_body = _capture_perf_body(prefill, prefill._make_run, 128, "hello", 1)
    assert generate_body.get("store") is False, generate_body
    assert prefill_body.get("store") is False, prefill_body


if __name__ == "__main__":
    tests = [
        test_eval_api_adapter_sets_store_false,
        test_eval_builders_set_store_false,
        test_perf_builders_set_store_false,
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
