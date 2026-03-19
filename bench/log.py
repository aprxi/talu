"""JSONL logging for evaluation runs with resume support."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Eval logs live under <project_root>/temp/evals/.
EVALS_DIR = Path(__file__).resolve().parent.parent / "temp" / "evals"


def eval_log_path(
    bench: str,
    model: str,
    samples: int | None = None,
    max_reasoning_tokens: int = 0,
) -> Path:
    """Stable log path for a bench + model + reasoning combination.

    Format: evals/<bench>_<model_slug>[_n<samples>][_r<budget>].jsonl
    Re-running the same combo appends to / resumes from this file.
    """
    slug = model.replace("/", "_").replace(" ", "_")
    name = f"{bench}_{slug}"
    if samples is not None:
        name += f"_n{samples}"
    if max_reasoning_tokens > 0:
        name += f"_r{max_reasoning_tokens}"
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    return EVALS_DIR / f"{name}.jsonl"


def load_completed(path: Path) -> tuple[set[tuple[str, int]], dict]:
    """Load completed entries from a JSONL log file.

    Returns:
        completed: set of (model_uri, index) pairs to skip.
        cached_stats: dict with cached_correct, token totals, and throughput lists.
    """
    completed: set[tuple[str, int]] = set()
    stats: dict = {
        "cached_correct": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "prefill_tok_s": [],
        "gen_tok_s": [],
        "meta": {},
    }
    if not path.exists():
        return completed, stats
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "meta":
                    stats["meta"] = rec
                    continue
                completed.add((rec["model"], rec["index"]))
                if rec.get("match", False):
                    stats["cached_correct"] += 1
                stats["total_input_tokens"] += rec.get("input_tokens", 0)
                stats["total_output_tokens"] += rec.get("output_tokens", 0)
                pt = rec.get("prefill_tok_s", 0)
                if pt > 0:
                    stats["prefill_tok_s"].append(pt)
                gt = rec.get("gen_tok_s", 0)
                if gt > 0:
                    stats["gen_tok_s"].append(gt)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed, stats


class EvalLogger:
    """Append-mode JSONL logger for per-question eval results."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a")  # noqa: SIM115

    def log(
        self,
        *,
        bench: str,
        index: int,
        question_hash: str,
        predicted: str | None,
        correct: str,
        model: str,
        raw_output: str,
        reasoning: str = "",
        response_output: list | None = None,
        question: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        prefill_tok_s: float = 0,
        gen_tok_s: float = 0,
        match: bool | None = None,
        extras: dict | None = None,
    ) -> None:
        record = {
            "bench": bench,
            "index": index,
            "question_hash": question_hash,
            "question": question,
            "predicted": predicted,
            "correct": correct,
            "match": match if match is not None else (predicted == correct),
            "model": model,
            "raw_output": raw_output,
            "reasoning": reasoning,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prefill_tok_s": round(prefill_tok_s, 1),
            "gen_tok_s": round(gen_tok_s, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if response_output is not None:
            record["response_output"] = response_output
        if extras:
            record.update(extras)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def write_meta(
        self,
        *,
        model_info: dict,
        max_reasoning_tokens: int = 0,
    ) -> None:
        """Write a meta record (once per log file, after first successful response)."""
        record = {
            "type": "meta",
            "model_info": model_info,
            "max_reasoning_tokens": max_reasoning_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
