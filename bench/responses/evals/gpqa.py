"""GPQA Diamond evaluation scenario for POST /v1/responses.

Registers responses/evals/gpqa.  Measures deep scientific reasoning
at graduate level (~198 questions in the Diamond subset).

Dataset: Idavidrein/gpqa, "gpqa_diamond" config, "train" split.
"""

from __future__ import annotations

import hashlib
import random

from scenario import Scenario
from responses.evals._runner import run_eval

_INSTRUCTIONS = (
    "You are taking a graduate-level multiple choice exam in science. "
    "Read the question and choices carefully. "
    "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not explain your reasoning."
)

_API_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
}


def _load_dataset(n: int | None = None):
    """Load GPQA Diamond samples with deterministically shuffled choices."""
    from datasets import load_dataset

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as exc:
        if "gated" in str(exc).lower():
            raise SystemExit(
                "GPQA is a gated dataset. Accept access at:\n"
                "  https://huggingface.co/datasets/Idavidrein/gpqa\n"
                "Then ensure your HF token is available (HF_HOME or huggingface-cli login)."
            ) from exc
        raise

    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        correct_text = row["Correct Answer"]
        choices = [
            correct_text,
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Deterministic shuffle seeded by question text to avoid position bias.
        rng = random.Random(row["Question"])
        rng.shuffle(choices)
        correct_letter = chr(65 + choices.index(correct_text))

        choices_str = "\n".join(
            f"{chr(65 + j)}. {c}" for j, c in enumerate(choices)
        )
        prompt = f"{row['Question']}\n\n{choices_str}"
        samples.append({
            "prompt": prompt,
            "correct": correct_letter,
            "question_hash": hashlib.sha256(row["Question"].encode()).hexdigest()[:16],
            "index": i,
        })
    return samples


def _build_body(sample: dict, uri: str, config: dict) -> dict:
    body: dict = {
        "model": uri,
        "input": sample["prompt"],
        "instructions": _INSTRUCTIONS,
        "stream": False,
        "store": False,
        "max_output_tokens": config.get("max_tokens", 64),
    }
    for cfg_key, api_key in _API_FIELDS.items():
        if cfg_key in config:
            body[api_key] = config[cfg_key]
    return body


class Gpqa(Scenario):
    name = "responses/evals/gpqa"
    description = "GPQA Diamond — graduate-level science reasoning."
    endpoint = "POST /v1/responses"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        print("  Loading GPQA Diamond dataset ...", flush=True)
        samples = _load_dataset(samples_n)
        print(f"  {len(samples)} samples loaded.\n", flush=True)

        return run_eval(
            bench_name="gpqa",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_build_body,
        )
