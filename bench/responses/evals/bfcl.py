"""BFCL evaluation scenario for POST /v1/responses.

Registers responses/evals/bfcl.  Measures function-calling accuracy on the
Berkeley Function Calling Leaderboard (v4) dataset using AST matching.

Phase 1 categories (no execution sandbox required):
  - Non-Live: simple_python (400), simple_java (100), simple_javascript (100),
    multiple (200), parallel (200), parallel_multiple (200)
  - Hallucination: irrelevance (875)
  - Live: live_simple (258), live_multiple (1037), live_parallel (16),
    live_parallel_multiple (24)

Metrics: overall accuracy, per-category accuracy.

Reference: Patil et al., "Gorilla: Large Language Model Connected with
Massive APIs" (arXiv:2305.15334).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from scenario import Scenario
from responses.evals._runner import run_eval
from responses.evals.bfcl_verify import match_simple, match_parallel, match_irrelevance

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "temp" / "bfcl"

_BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

# Categories to evaluate: (file_stem, scoring_mode).
# scoring_mode: "simple" (1 call), "parallel" (N calls, unordered), "irrelevance" (no calls).
_CATEGORIES: list[tuple[str, str]] = [
    # Non-Live.
    ("BFCL_v4_simple_python", "simple"),
    ("BFCL_v4_simple_java", "simple"),
    ("BFCL_v4_simple_javascript", "simple"),
    ("BFCL_v4_multiple", "simple"),
    ("BFCL_v4_parallel", "parallel"),
    ("BFCL_v4_parallel_multiple", "parallel"),
    # Hallucination.
    ("BFCL_v4_irrelevance", "irrelevance"),
    # Live.
    ("BFCL_v4_live_simple", "simple"),
    ("BFCL_v4_live_multiple", "simple"),
    ("BFCL_v4_live_parallel", "parallel"),
    ("BFCL_v4_live_parallel_multiple", "parallel"),
]

_API_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def _download(url: str, dest: Path) -> None:
    """Download a file if not already cached."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _convert_params(params: dict) -> dict:
    """Convert BFCL parameter schema to OpenAI-compatible JSON Schema.

    BFCL uses ``"type": "dict"`` where OpenAI expects ``"type": "object"``.
    Recursively fixes nested schemas.
    """
    out = dict(params)
    if out.get("type") == "dict":
        out["type"] = "object"
    props = out.get("properties")
    if isinstance(props, dict):
        new_props = {}
        for k, v in props.items():
            if isinstance(v, dict):
                new_props[k] = _convert_params(v)
            else:
                new_props[k] = v
        out["properties"] = new_props
    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = _convert_params(items)
    return out


def _sanitize_name(name: str) -> str:
    """Sanitize a BFCL function name to match ``^[a-zA-Z0-9_-]{1,64}$``."""
    return name.replace(".", "_")


def _convert_tools(functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to talu ``tools`` format.

    Talu expects: ``{"type": "function", "name": "...", "description": "...", "parameters": {...}}``.
    Names containing dots are converted to underscores (e.g. ``math.factorial`` → ``math_factorial``).
    """
    tools = []
    for func in functions:
        tool: dict = {
            "type": "function",
            "name": _sanitize_name(func["name"]),
            "description": func.get("description", ""),
            "parameters": _convert_params(func.get("parameters", {})),
        }
        tools.append(tool)
    return tools


def _load_dataset(
    categories: list[tuple[str, str]] | None = None,
    n: int | None = None,
) -> list[dict]:
    """Download and load BFCL samples across all Phase 1 categories.

    Args:
        categories: Override category list (for testing).
        n: Limit total samples (distributed across categories proportionally).
    """
    if categories is None:
        categories = _CATEGORIES

    all_samples: list[dict] = []
    global_idx = 0

    for file_stem, scoring_mode in categories:
        # Download question file.
        q_url = f"{_BASE_URL}/{file_stem}.json"
        q_path = _CACHE_DIR / f"{file_stem}.json"
        _download(q_url, q_path)

        questions = _load_jsonl(q_path)

        # Download ground truth (not available for irrelevance — score by absence).
        gt_by_id: dict[str, list[dict]] = {}
        if scoring_mode != "irrelevance":
            gt_url = f"{_BASE_URL}/possible_answer/{file_stem}.json"
            gt_path = _CACHE_DIR / f"{file_stem}_answer.json"
            _download(gt_url, gt_path)
            for rec in _load_jsonl(gt_path):
                gt_by_id[rec["id"]] = rec.get("ground_truth", [])

        for q in questions:
            qid = q["id"]
            # Extract prompt from question field: [[{"role": "user", "content": "..."}]].
            turns = q.get("question", [[]])
            if turns and turns[0]:
                prompt = turns[0][0].get("content", "")
            else:
                prompt = ""

            functions = q.get("function", [])
            ground_truth = gt_by_id.get(qid, [])

            # Short category label from file stem (e.g. "simple_python", "live_multiple").
            cat_label = file_stem.removeprefix("BFCL_v4_")

            all_samples.append({
                "prompt": prompt,
                "correct": "",
                "question_hash": hashlib.sha256(f"{qid}:{prompt}".encode()).hexdigest()[:16],
                "index": global_idx,
                "bfcl_id": qid,
                "tools": functions,
                "ground_truth": ground_truth,
                "category": cat_label,
                "scoring_mode": scoring_mode,
            })
            global_idx += 1

    # Limit samples if requested (take first N to keep category proportions).
    if n is not None and n < len(all_samples):
        all_samples = all_samples[:n]

    return all_samples


def _build_body(sample: dict, uri: str, config: dict) -> dict:
    """Build canonical request dict. API format translation is handled by _api.py."""
    canonical: dict = {
        "model": uri,
        "input": sample["prompt"],
    }

    # Convert and attach tools.
    functions = sample.get("tools", [])
    if functions:
        canonical["tools"] = _convert_tools(functions)
        if sample.get("scoring_mode") == "irrelevance":
            canonical["tool_choice"] = "auto"
        else:
            canonical["tool_choice"] = "required"

    if "max_tokens" in config:
        canonical["max_output_tokens"] = config["max_tokens"]
    mrt = 0
    if "max_reasoning_tokens" in config and config.get("max_reasoning_tokens") is not None:
        mrt = int(config["max_reasoning_tokens"])
        canonical["max_reasoning_tokens"] = mrt
    # Tool calls need a completion budget. For the responses API, this is
    # derived from max_output_tokens. For completions API, max_tokens is
    # the hard cap. Default to reasoning + 512 tokens for the tool output.
    if "max_completion_tokens" not in config and "max_tokens" not in config:
        canonical["max_completion_tokens"] = mrt + 512

    for key in _API_FIELDS:
        if key in config:
            canonical[key] = config[key]

    return canonical


def _extract_tool_calls(events: list[dict]) -> list[dict]:
    """Extract function_call items from response events.

    Handles both v1/responses format (output items with type=function_call)
    and v1/chat/completions format (choices[0].message.tool_calls).

    Returns list of {"name": str, "arguments": dict}.
    """
    calls: list[dict] = []
    for ev in events:
        if ev.get("event") not in ("response.completed", "response.incomplete"):
            continue
        resp = ev.get("data", {}).get("response", {})

        # v1/responses format: output items
        for item in resp.get("output", []):
            if item.get("type") == "function_call":
                name = item.get("name", "")
                args_raw = item.get("arguments", "{}")
                calls.append({"name": name, "arguments": _parse_args(args_raw)})

        # v1/chat/completions format: choices[0].message.tool_calls
        if not calls:
            for choice in resp.get("choices", []):
                msg = choice.get("message", {})
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_raw = func.get("arguments", "{}")
                    calls.append({"name": name, "arguments": _parse_args(args_raw)})
    return calls


def _parse_args(args_raw) -> dict:
    """Parse tool call arguments from string or dict."""
    if isinstance(args_raw, str):
        try:
            return json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(args_raw, dict):
        return args_raw
    return {}


def _extract_output_summary(events: list[dict]) -> str:
    """Summarize output item types and sizes from response events (for debugging)."""
    parts: list[str] = []
    for ev in events:
        if ev.get("event") not in ("response.completed", "response.incomplete"):
            continue
        resp = ev.get("data", {}).get("response", {})

        # v1/responses format
        for item in resp.get("output", []):
            itype = item.get("type", "?")
            if itype == "function_call":
                name = item.get("name", "")
                args = item.get("arguments", "")
                parts.append(f"function_call({name}, {len(str(args))}b)")
            elif itype == "message":
                for part in item.get("content", []):
                    ptype = part.get("type", "?")
                    text = part.get("text", "")
                    parts.append(f"{ptype}({len(text)}b)")
            elif itype == "reasoning":
                for s in item.get("summary", []):
                    text = s.get("text", "")
                    parts.append(f"reasoning({len(text)}b)")
            else:
                parts.append(itype)

        # v1/chat/completions format
        if not parts:
            for choice in resp.get("choices", []):
                msg = choice.get("message", {})
                content = msg.get("content", "")
                if content:
                    parts.append(f"text({len(content)}b)")
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    parts.append(f"function_call({name}, {len(str(args))}b)")

    return ", ".join(parts) if parts else "empty"


def _score_fn(raw: str, sample: dict, events: list[dict] | None = None) -> dict:
    """Score a BFCL response by comparing tool calls against ground truth."""
    events = events or []
    actual_calls = _extract_tool_calls(events)
    scoring_mode = sample.get("scoring_mode", "simple")
    ground_truth = sample.get("ground_truth", [])
    func_schemas = sample.get("tools", [])

    if scoring_mode == "irrelevance":
        is_correct = match_irrelevance(actual_calls)
    elif scoring_mode == "parallel":
        is_correct = match_parallel(actual_calls, ground_truth, func_schemas)
    else:
        # simple / multiple
        is_correct = match_simple(actual_calls, ground_truth, func_schemas)

    output_summary = _extract_output_summary(events)

    return {
        "predicted": json.dumps([c["name"] for c in actual_calls]),
        "is_correct": is_correct,
        "category": sample.get("category", ""),
        "n_calls": len(actual_calls),
        "n_expected": len(ground_truth),
        "output_types": output_summary,
    }


def _read_bfcl_records(path: Path) -> list[dict]:
    """Read BFCL per-question records from a JSONL log (skips meta records)."""
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


class Bfcl(Scenario):
    name = "responses/evals/bfcl"
    description = "BFCL — function-calling accuracy (AST matching)."
    endpoint = "POST /v1/responses"

    def prepare_config(self, config: dict) -> None:
        # BFCL tool calls are fragile with thinking disabled on small models.
        # Use a modest default budget unless the user overrides it.
        if "max_reasoning_tokens" not in config:
            config["max_reasoning_tokens"] = 256

        # Structured tool calls are sensitive to sampling noise. If config
        # still matches the generic bench defaults, switch BFCL to a stable
        # deterministic profile.
        if (
            float(config.get("temperature", 1.0)) == 1.0
            and float(config.get("top_p", 0.95)) == 0.95
            and int(config.get("top_k", 20)) == 20
            and float(config.get("presence_penalty", 1.5)) == 1.5
        ):
            config["temperature"] = 0.0
            config["top_p"] = 1.0
            config["top_k"] = 1
            config["presence_penalty"] = 0.0

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        print("  Loading BFCL v4 dataset ...", flush=True)
        samples = _load_dataset(n=samples_n)

        # Count per category.
        from collections import Counter
        cat_counts = Counter(s["category"] for s in samples)
        cats_str = ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items()))
        print(f"  {len(samples)} samples ({cats_str}).\n", flush=True)

        results = run_eval(
            bench_name="bfcl",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_build_body,
            score_fn=_score_fn,
            completions=config.get("_completions", False),
        )

        # Post-aggregate: per-category accuracy from JSONL logs.
        from log import eval_log_path
        for r in results:
            mrt = int(r.get("max_reasoning_tokens", 0))
            log_path = eval_log_path(
                "bfcl", r.get("model_uri", r["model"]),
                samples_n, mrt,
                endpoint=config.get("_endpoint"),
                session_id=config.get("_session_id"),
            )
            records = _read_bfcl_records(log_path)
            scored = [rec for rec in records if "category" in rec]
            if not scored:
                continue

            # Per-category stats.
            cat_stats: dict[str, dict[str, int]] = {}
            for rec in scored:
                cat = rec.get("category", "unknown")
                if cat not in cat_stats:
                    cat_stats[cat] = {"correct": 0, "total": 0}
                cat_stats[cat]["total"] += 1
                if rec.get("match"):
                    cat_stats[cat]["correct"] += 1

            # Store per-category accuracy.
            cat_accs: dict[str, float] = {}
            for cat, stats in sorted(cat_stats.items()):
                acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
                cat_accs[cat] = round(acc, 1)

            r["category_accuracy"] = cat_accs

        return results
