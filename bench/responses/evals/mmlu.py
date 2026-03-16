"""MMLU evaluation scenario for POST /v1/responses.

Registers responses/evals/mmlu.  Measures broad knowledge + reasoning
accuracy across 50+ subjects (~15k questions).

Dataset: cais/mmlu, "all" config, "test" split.
"""

from __future__ import annotations

import hashlib

from scenario import Scenario
from responses.evals._runner import run_eval

_ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

_INSTRUCTIONS = (
    "You are taking a multiple choice exam. "
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
    """Load MMLU test samples. Returns list of dicts."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        choices = "\n".join(
            f"{chr(65 + j)}. {c}" for j, c in enumerate(row["choices"])
        )
        prompt = f"{row['question']}\n\n{choices}"
        samples.append({
            "prompt": prompt,
            "correct": _ANSWER_MAP[row["answer"]],
            "question_hash": hashlib.sha256(row["question"].encode()).hexdigest()[:16],
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
        "max_output_tokens": config.get("max_tokens", 4096),
    }
    for cfg_key, api_key in _API_FIELDS.items():
        if cfg_key in config:
            body[api_key] = config[cfg_key]
    if "reasoning_effort" in config:
        body["reasoning"] = {"effort": config["reasoning_effort"]}
    return body


class Mmlu(Scenario):
    name = "responses/evals/mmlu"
    description = "MMLU — broad knowledge + reasoning accuracy."
    endpoint = "POST /v1/responses"

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        print("  Loading MMLU dataset ...", flush=True)
        samples = _load_dataset(samples_n)
        print(f"  {len(samples)} samples loaded.\n", flush=True)

        return run_eval(
            bench_name="mmlu",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_build_body,
        )
