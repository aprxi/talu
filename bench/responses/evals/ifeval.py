"""IFEval evaluation scenario for POST /v1/responses.

Registers responses/evals/ifeval.  Measures instruction-following accuracy
using 25 verifiable instruction types (~541 prompts).

Dataset: google/IFEval, "train" split (single split).
Metrics: prompt-level accuracy, instruction-level accuracy (strict).

Reference: Zhou et al., "Instruction-Following Evaluation for Large
Language Models" (arXiv:2311.07911, 2023).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scenario import Scenario
from responses.evals._runner import run_eval
from responses.evals.ifeval_verify import evaluate_strict

def _read_ifeval_records(path: Path) -> list[dict]:
    """Read IFEval per-question records from a JSONL log (skips meta records)."""
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "meta":
                    continue
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


_API_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
}


def _load_dataset(n: int | None = None):
    """Load IFEval samples."""
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        # kwargs may be stored as JSON strings — parse if needed.
        kwargs_raw = row["kwargs"]
        kwargs_list = []
        for kw in kwargs_raw:
            if isinstance(kw, str):
                kw = json.loads(kw)
            kwargs_list.append(kw)

        samples.append({
            "prompt": row["prompt"],
            "correct": "",  # Not MCQ — scoring uses instruction checkers.
            "question_hash": hashlib.sha256(row["prompt"].encode()).hexdigest()[:16],
            "index": i,
            "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": kwargs_list,
        })
    return samples


def _build_body(sample: dict, uri: str, config: dict) -> dict:
    body: dict = {
        "model": uri,
        "input": sample["prompt"],
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


def _score_fn(raw: str, sample: dict, events: list[dict] | None = None) -> dict:
    """Score a single IFEval response using instruction checkers."""
    instruction_ids = sample["instruction_id_list"]
    kwargs_list = sample["kwargs"]
    prompt = sample.get("prompt", "")

    strict = evaluate_strict(raw, instruction_ids, kwargs_list, prompt)

    return {
        "predicted": "",
        "is_correct": all(strict),
        "strict_results": strict,
        "prompt_strict": all(strict),
    }


class Ifeval(Scenario):
    name = "responses/evals/ifeval"
    description = "IFEval — instruction-following accuracy."
    endpoint = "POST /v1/responses"

    def prepare_config(self, config: dict) -> None:
        # Default to non-thinking mode.
        if "max_reasoning_tokens" not in config:
            config["max_reasoning_tokens"] = 0

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        print("  Loading IFEval dataset ...", flush=True)
        samples = _load_dataset(samples_n)
        n_instructions = sum(len(s["instruction_id_list"]) for s in samples)
        print(f"  {len(samples)} prompts, {n_instructions} instructions.\n", flush=True)

        results = run_eval(
            bench_name="ifeval",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_build_body,
            score_fn=_score_fn,
        )

        # Post-aggregate the 4 IFEval metrics from JSONL logs (covers
        # both fresh and cached/resumed entries).
        from log import eval_log_path
        for r in results:
            mrt = int(r.get("max_reasoning_tokens", 0))
            log_path = eval_log_path(
                "ifeval", r.get("model_uri", r["model"]),
                samples_n, mrt,
                endpoint=config.get("_endpoint"),
                session_id=config.get("_session_id"),
            )
            records = _read_ifeval_records(log_path)
            # Only count records that have IFEval extras (skip stale cached entries).
            scored = [rec for rec in records if "strict_results" in rec]
            if not scored:
                continue

            prompt_strict_n = sum(1 for rec in scored if rec.get("prompt_strict"))
            all_strict = [b for rec in scored for b in rec.get("strict_results", [])]

            n = len(scored)
            r["prompt_strict_acc"] = prompt_strict_n / n * 100 if n else 0
            r["inst_strict_acc"] = (
                sum(all_strict) / len(all_strict) * 100 if all_strict else 0
            )

        return results
