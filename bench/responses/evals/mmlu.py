"""MMLU evaluation scenario for POST /v1/responses.

Registers responses/evals/mmlu.  Measures broad knowledge + reasoning
accuracy across 57 subjects (~14k questions).

Dataset: cais/mmlu, "all" config, "test" split.

When --samples N is given, automatically selects subjects and distributes
questions evenly (~20 per subject, minimum 5 subjects, deterministic).
Without --samples, runs the full dataset.
"""

from __future__ import annotations

import hashlib
import math
import random

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

# All 57 MMLU subjects in priority order.  Broadly useful STEM subjects
# first so small --samples values get good signal, then broadening out.
_SUBJECT_PRIORITY = [
    # Core STEM (strong general signal).
    "college_biology",
    "college_physics",
    "college_computer_science",
    "high_school_mathematics",
    "machine_learning",
    # Extended STEM.
    "college_chemistry",
    "high_school_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "abstract_algebra",
    "high_school_statistics",
    "econometrics",
    "anatomy",
    "electrical_engineering",
    "college_mathematics",
    "college_medicine",
    "elementary_mathematics",
    "astronomy",
    "conceptual_physics",
    # Logic & reasoning.
    "computer_security",
    "formal_logic",
    "logical_fallacies",
    # Health & medicine.
    "clinical_knowledge",
    "medical_genetics",
    "virology",
    "nutrition",
    "human_aging",
    "human_sexuality",
    # Humanities.
    "philosophy",
    "moral_disputes",
    "moral_scenarios",
    "world_religions",
    "prehistory",
    # Business & social sciences.
    "business_ethics",
    "management",
    "marketing",
    "public_relations",
    "sociology",
    "high_school_psychology",
    # Professional.
    "professional_medicine",
    "professional_psychology",
    "professional_accounting",
    "professional_law",
    # Geography & government.
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    # History.
    "high_school_us_history",
    "high_school_world_history",
    "high_school_european_history",
    # Law.
    "international_law",
    "jurisprudence",
    "security_studies",
    # Broad.
    "global_facts",
    "miscellaneous",
]

_MIN_SUBJECTS = 5
_TARGET_PER_SUBJECT = 20


def _load_dataset(samples_n: int | None = None):
    """Load MMLU test samples.

    When *samples_n* is given, automatically select subjects and distribute
    questions evenly (~20 per subject, min 5 subjects, deterministic seed=42).
    Without it, load the full dataset.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")

    if samples_n is None:
        # Full dataset.
        return _format_samples(ds)

    # Group row indices by subject.
    by_subject: dict[str, list[int]] = {}
    for idx in range(len(ds)):
        by_subject.setdefault(ds[idx]["subject"], []).append(idx)

    # Build ordered subject list: priority list first, then any extras.
    all_subjects = list(by_subject.keys())
    ordered = [s for s in _SUBJECT_PRIORITY if s in by_subject]
    for s in sorted(all_subjects):
        if s not in ordered:
            ordered.append(s)

    # Choose how many subjects.
    n_subjects = math.ceil(samples_n / _TARGET_PER_SUBJECT)
    n_subjects = max(_MIN_SUBJECTS, min(len(ordered), n_subjects))
    subjects = ordered[:n_subjects]

    # Distribute samples evenly.
    base = samples_n // n_subjects
    remainder = samples_n % n_subjects

    # Sample deterministically within each subject.
    selected: list[int] = []
    for i, subj in enumerate(subjects):
        pool = by_subject[subj]
        k = min(base + (1 if i < remainder else 0), len(pool))
        rng = random.Random(f"42:{subj}")
        rng.shuffle(pool)
        selected.extend(sorted(pool[:k]))

    ds = ds.select(selected)
    return _format_samples(ds)


def _format_samples(ds) -> list[dict]:
    """Convert dataset rows to sample dicts."""
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
        "max_completion_tokens": 1,
        "stream": False,
        "store": False,
    }
    if "max_tokens" in config:
        body["max_output_tokens"] = config["max_tokens"]
    mrt = int(config.get("max_reasoning_tokens", 0))
    body["max_reasoning_tokens"] = mrt
    for cfg_key, api_key in _API_FIELDS.items():
        if cfg_key in config:
            body[api_key] = config[cfg_key]
    return body


class Mmlu(Scenario):
    name = "responses/evals/mmlu"
    description = "MMLU — broad knowledge + reasoning accuracy."
    endpoint = "POST /v1/responses"

    def prepare_config(self, config: dict) -> None:
        # Default to non-thinking mode (user can override with --set max_reasoning_tokens=N).
        if "max_reasoning_tokens" not in config:
            config["max_reasoning_tokens"] = 0

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        if samples_n is not None:
            n_subjects = math.ceil(samples_n / _TARGET_PER_SUBJECT)
            n_subjects = max(_MIN_SUBJECTS, min(len(_SUBJECT_PRIORITY), n_subjects))
            label = f"MMLU ({n_subjects} subjects, ~{samples_n // n_subjects}q each, {samples_n} total)"
        else:
            label = "MMLU (full dataset)"
        print(f"  Loading {label} ...", flush=True)

        samples = _load_dataset(samples_n)
        print(f"  {len(samples)} samples loaded.\n", flush=True)

        return run_eval(
            bench_name="mmlu",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_build_body,
        )
