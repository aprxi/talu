from __future__ import annotations

from pathlib import Path
import time

from talu import Chat, GenerationConfig

MODEL_URI = Path("models/Qwen/Qwen3-0.6B-GAF4")


def _assert_model_available() -> None:
    if not MODEL_URI.exists():
        raise SystemExit(f"Model path missing: {MODEL_URI}")


def _tps_from_response(response, fallback_tokens: int, elapsed_s: float | None = None) -> float:
    if response.timings and response.timings.generation_ms > 0:
        return response.timings.tokens_per_second
    if elapsed_s and elapsed_s > 0:
        tokens = response.usage.completion_tokens if response.usage else fallback_tokens
        return tokens / elapsed_s
    return 0.0


def main() -> None:
    _assert_model_available()
    chat = Chat(MODEL_URI.as_posix())
    try:
        baseline_start = time.perf_counter()
        baseline = chat.send(
            "Write a short technical paragraph about JSON schemas.",
            config=GenerationConfig(max_tokens=200, temperature=0.7),
        )
        baseline_elapsed = time.perf_counter() - baseline_start
        baseline_tps = _tps_from_response(baseline, 200, baseline_elapsed)

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["id", "name", "value"],
            },
        }
        prompt = (
            "Return a JSON array of objects with id, name, and value fields. "
            "Return only JSON."
        )
        constrained_start = time.perf_counter()
        constrained = chat.send(
            prompt,
            config=GenerationConfig(max_tokens=200, temperature=0.7),
            response_format=schema,
        )
        constrained_elapsed = time.perf_counter() - constrained_start
        constrained_tps = _tps_from_response(constrained, 200, constrained_elapsed)

        print("baseline_tps:", f"{baseline_tps:.2f}")
        print("constrained_tps:", f"{constrained_tps:.2f}")
        if baseline_tps > 0:
            ratio = constrained_tps / baseline_tps
            print("ratio:", f"{ratio:.2%}")
            if ratio < 0.85:
                print("WARNING: constrained TPS below 85% baseline. Investigate core/src/grammar/engine.zig")
    finally:
        del chat


if __name__ == "__main__":
    main()
