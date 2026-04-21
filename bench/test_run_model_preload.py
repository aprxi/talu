"""Regression checks for per-run server model preloading in bench/run.py.

Run:
    python bench/test_run_model_preload.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as bench_run


class _FakeScenario:
    def prepare_config(self, config: dict) -> None:
        pass

    def server_args(self, config: dict) -> list[str]:
        return ["--log-format", "json"]

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        return [{
            "model": config["model_uri"][0],
            "scheme": config["precision"][0],
            "model_uri": config["model_uri"][0],
            "correct_count": 1,
            "total": 1,
            "accuracy": 100.0,
            "results": [],
            "bench": "fake",
            "errors": 0,
            "avg_gen_tok_s": 0.0,
            "avg_prefill_tok_s": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_reasoning_tokens": 0,
            "model_info": {},
        }]


class _FakeServer:
    instances: list["_FakeServer"] = []

    def __init__(self, *, port: int, extra_args: list[str] | None = None, env: dict[str, str] | None = None, **_: object) -> None:
        self.port = port
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})
        self.pid = 12345
        self.binary = Path("/tmp/fake-talu")
        _FakeServer.instances.append(self)

    def start(self, timeout: float = 30) -> None:
        _ = timeout

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:9999"

    def stop(self) -> None:
        pass


def _make_args(*, precision: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="run",
        scenario="responses/evals/fake",
        config=None,
        set=[f"model_uri=Qwen/Qwen3.5-27B-NVFP4", f"precision={precision}"],
        env=[],
        resume="session123",
        samples=1,
        batched=8,
        rounds=1,
        endpoint=None,
        port=18258,
    )


def _run_cmd(args: argparse.Namespace) -> _FakeServer:
    original_get_scenario = bench_run.get_scenario
    original_load_config = bench_run.load_config
    original_talu_server = bench_run.TaluServer
    original_get_version = bench_run._get_version
    original_detect_hardware = bench_run._detect_hardware
    original_print_expanded_cmd = bench_run._print_expanded_cmd
    original_print_eval_report = bench_run.print_eval_report

    precision = next(
        (item.split("=", 1)[1] for item in args.set if item.startswith("precision=")),
        "original",
    )
    try:
        _FakeServer.instances = []
        bench_run.get_scenario = lambda name: _FakeScenario if name == "responses/evals/fake" else None
        bench_run.load_config = lambda *_args, **_kwargs: {
            "model_uri": ["Qwen/Qwen3.5-27B-NVFP4"],
            "precision": [precision],
            "env": {},
            "seed": 42,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
        }
        bench_run.TaluServer = _FakeServer
        bench_run._get_version = lambda _binary: "test-version"
        bench_run._detect_hardware = lambda _env: "test-hw"
        bench_run._print_expanded_cmd = lambda _args, _config: None
        bench_run.print_eval_report = lambda *_args, **_kwargs: None

        bench_run.cmd_run(args)
        assert len(_FakeServer.instances) >= 2
        return _FakeServer.instances[-1]
    finally:
        bench_run.get_scenario = original_get_scenario
        bench_run.load_config = original_load_config
        bench_run.TaluServer = original_talu_server
        bench_run._get_version = original_get_version
        bench_run._detect_hardware = original_detect_hardware
        bench_run._print_expanded_cmd = original_print_expanded_cmd
        bench_run.print_eval_report = original_print_eval_report


def test_with_server_model_arg_appends_model() -> None:
    args = bench_run._with_server_model_arg(["--log-format", "json"], "Qwen/Qwen3.5-27B-NVFP4")
    assert args == ["--log-format", "json", "--model", "Qwen/Qwen3.5-27B-NVFP4"], args


def test_with_server_model_arg_preserves_existing_model_flag() -> None:
    args = bench_run._with_server_model_arg(
        ["--log-format", "json", "--model", "Already/Selected"],
        "Qwen/Qwen3.5-27B-NVFP4",
    )
    assert args == ["--log-format", "json", "--model", "Already/Selected"], args


def test_cmd_run_preloads_original_model_uri() -> None:
    server = _run_cmd(_make_args(precision="original"))
    assert server.extra_args == [
        "--log-format", "json",
        "--model", "Qwen/Qwen3.5-27B-NVFP4",
    ], server.extra_args


def test_cmd_run_preloads_quantized_model_uri() -> None:
    server = _run_cmd(_make_args(precision="TQ4"))
    assert server.extra_args == [
        "--log-format", "json",
        "--model", "Qwen/Qwen3.5-27B-NVFP4-TQ4",
    ], server.extra_args


if __name__ == "__main__":
    tests = [
        test_with_server_model_arg_appends_model,
        test_with_server_model_arg_preserves_existing_model_flag,
        test_cmd_run_preloads_original_model_uri,
        test_cmd_run_preloads_quantized_model_uri,
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
