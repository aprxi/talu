"""Scenario base class, registry, config loading, and HTTP helpers.

Scenario naming:  <endpoint>/<kind>/<name>   e.g. responses/perf/hello
Config files:     <endpoint>/conf/<config_name>.json

Config model fields:
    model_uri:   List of base model IDs (or comma-separated string via CLI).
                 e.g. ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-2B"]
    precision:   List of precision schemes (or comma-separated string via CLI).
                 e.g. ["original", "GAF4"]
                 "original" = unmodified model (no URI suffix).
                 Every model × precision combination is benchmarked.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

_SCENARIOS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, type[Scenario]] = {}


def list_scenarios() -> dict[str, list[type[Scenario]]]:
    """Return scenarios grouped by endpoint prefix."""
    groups: dict[str, list[type[Scenario]]] = {}
    for cls in _SCENARIOS.values():
        prefix = cls.name.rsplit("/", 1)[0] if "/" in cls.name else cls.name
        groups.setdefault(prefix, []).append(cls)
    for v in groups.values():
        v.sort(key=lambda c: c.name)
    return dict(sorted(groups.items()))


def get_scenario(name: str) -> type[Scenario] | None:
    return _SCENARIOS.get(name)


def scenario_names() -> list[str]:
    return sorted(_SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Sampling presets (mirrors core/src/models/sampling_presets.zig)
# ---------------------------------------------------------------------------

SAMPLING_PRESETS: dict[str, dict[str, float | int]] = {
    "general":       {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    "coding":        {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    "instruct":      {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "presence_penalty": 1.5},
    "deterministic": {"temperature": 0.0, "top_p": 1.0,  "top_k": 1,  "presence_penalty": 0.0},
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "model_uri": ["Qwen/Qwen3.5-0.8B"],
    "precision": ["original", "GAF8", "GAF4"],
    "env": {},
    "streaming": False,
    "seed": 42,
    **SAMPLING_PRESETS["general"],
}


def load_config(
    scenario_name: str,
    config_name: str | None,
    overrides: list[str] | None = None,
    env_overrides: list[str] | None = None,
) -> dict:
    """Load config JSON, then apply CLI --set and --env overrides.

    Looks in scenarios/<group>/conf/<config_name>.json.
    Falls back to built-in defaults if no config specified.
    """
    config = dict(_DEFAULT_CONFIG)
    # Deep-copy env so mutations don't affect the default.
    config["env"] = dict(config.get("env", {}))

    if config_name is not None:
        group = scenario_name.split("/")[0]
        conf_path = _SCENARIOS_DIR / group / "conf" / f"{config_name}.json"

        if not conf_path.exists():
            available = list_configs(scenario_name)
            avail_str = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"config not found: {conf_path}\n"
                f"Available configs for {group}/: {avail_str}"
            )

        with open(conf_path) as f:
            file_overrides = json.load(f)
        # Merge env from file into config env (don't replace the whole dict).
        if "env" in file_overrides:
            config["env"].update(file_overrides.pop("env"))
        config.update(file_overrides)

    # Apply CLI --set overrides (track which keys were explicitly set).
    explicit_keys: set[str] = set()
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set requires key=value format, got: {item!r}")
        key, val_str = item.split("=", 1)
        config[key] = _parse_value(val_str)
        explicit_keys.add(key)

    # Resolve preset: apply preset values for keys not explicitly overridden.
    preset_name = config.pop("preset", None)
    if isinstance(preset_name, str):
        if preset_name not in SAMPLING_PRESETS:
            avail = ", ".join(sorted(SAMPLING_PRESETS))
            raise ValueError(
                f"unknown preset {preset_name!r} (available: {avail})"
            )
        for k, v in SAMPLING_PRESETS[preset_name].items():
            if k not in explicit_keys:
                config[k] = v

    # Apply CLI --env overrides.
    for item in env_overrides or []:
        if "=" not in item:
            raise ValueError(f"--env requires KEY=VALUE format, got: {item!r}")
        key, val_str = item.split("=", 1)
        config["env"][key] = val_str

    # Normalize list fields: comma-separated strings → lists.
    for key in ("model_uri", "precision"):
        val = config.get(key)
        if isinstance(val, str):
            config[key] = [v.strip() for v in val.split(",")]

    # Normalize max_reasoning_tokens: comma-separated → list of ints.
    mrt = config.get("max_reasoning_tokens")
    if isinstance(mrt, str):
        config["max_reasoning_tokens"] = [int(v.strip()) for v in mrt.split(",")]
    elif isinstance(mrt, (int, float)):
        config["max_reasoning_tokens"] = [int(mrt)]

    return config


def _parse_value(s: str) -> object:
    """Parse a CLI value: try JSON first, fall back to string."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def list_configs(scenario_name: str) -> list[str]:
    """List available config names for a scenario's endpoint group."""
    group = scenario_name.split("/")[0]
    conf_dir = _SCENARIOS_DIR / group / "conf"
    if not conf_dir.is_dir():
        return []
    return sorted(p.stem for p in conf_dir.glob("*.json"))


def model_uri(base: str, quant: str | None = None) -> str:
    """Build the full model URI: base or base-QUANT."""
    return f"{base}-{quant}" if quant else base


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Scenario:
    """Base class for benchmark scenarios.

    Subclass attributes:
        name:        Slash-separated identifier, e.g. "responses/hello".
        description: One-line summary shown in `list`.
        endpoint:    The talu serve endpoint being benchmarked.
    """

    name: str = ""
    description: str = ""
    endpoint: str = ""
    family: str = ""
    report_type: str = "responses"
    requires_storage: bool = False
    uses_model_matrix: bool = True

    def server_args(self, config: dict) -> list[str]:
        """Extra CLI args for talu serve, derived from config."""
        return []

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        """Execute the scenario. Return list of result dicts (one per round)."""
        raise NotImplementedError

    def __init_subclass__(cls, **kw: object) -> None:
        super().__init_subclass__(**kw)
        if cls.name:
            _SCENARIOS[cls.name] = cls


# ---------------------------------------------------------------------------
# HTTP helpers (zero dependencies)
# ---------------------------------------------------------------------------

def http_post_stream(
    url: str,
    body: dict,
    *,
    timeout: float = 120,
) -> tuple[list[dict], float]:
    """POST JSON, collect SSE events. Returns (events, wall_seconds)."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    addr = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"

    payload = json.dumps(body).encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((addr, port))

    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {addr}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + payload

    sock.sendall(request)

    t0 = time.monotonic()
    chunks: list[bytes] = []
    while True:
        try:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break
    wall_s = time.monotonic() - t0
    sock.close()

    raw = b"".join(chunks).decode(errors="replace")

    if "\r\n\r\n" in raw:
        header_str, body_str = raw.split("\r\n\r\n", 1)
    else:
        header_str, body_str = "", raw

    # Check for HTTP errors.
    status_line = header_str.split("\r\n", 1)[0] if header_str else ""
    if status_line and " 200 " not in status_line:
        print(f"\n    HTTP error: {status_line}", flush=True)
        if body_str.strip():
            # Try to parse JSON error body.
            try:
                err = json.loads(body_str.strip())
                print(f"    {err}", flush=True)
            except json.JSONDecodeError:
                print(f"    {body_str[:200]}", flush=True)

    events: list[dict] = []
    current_event = ""
    for line in body_str.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data = line[6:]
            try:
                parsed_data = json.loads(data)
                events.append({"event": current_event, "data": parsed_data})
            except json.JSONDecodeError:
                pass
        elif line == "":
            current_event = ""

    # Non-streaming: body is a single JSON response, not SSE events.
    if not events and body_str.strip():
        try:
            resp = json.loads(body_str.strip())
            status = resp.get("status", "completed")
            event_type = "response.completed" if status == "completed" else "response.incomplete"
            events.append({"event": event_type, "data": {"response": resp}})
        except json.JSONDecodeError:
            pass

    return events, wall_s


def extract_generation_metrics(events: list[dict]) -> dict:
    """Extract tok/s, output_tokens, decode time from response SSE events."""
    last_tokens_gen = 0
    last_elapsed_ms = 0.0
    input_tokens = 0
    output_tokens = 0
    prefill_ms = 0
    generation_ms = 0.0
    ttft_ms = 0
    model_info: dict = {}

    for ev in events:
        data = ev["data"]
        if ev["event"] in ("response.output_text.delta", "response.reasoning.delta"):
            tg = data.get("tokens_generated", 0)
            em = data.get("elapsed_ms", 0)
            if tg > 0 and em > 0:
                last_tokens_gen = tg
                last_elapsed_ms = em
        elif ev["event"] in ("response.completed", "response.incomplete"):
            resp = data.get("response", {})
            usage = resp.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            prefill_ms = usage.get("prefill_ms", 0)
            generation_ms = usage.get("generation_ms", 0)
            ttft_ms = usage.get("ttft_ms", 0)
            model_info = resp.get("model_info", {})
        elif ev["event"] == "error":
            err = data.get("error", data)
            msg = err.get("message", err) if isinstance(err, dict) else err
            print(f"\n    ERROR: {msg}", flush=True)
        elif ev["event"] == "response.failed":
            resp = data.get("response", {})
            err = resp.get("error", {})
            msg = err.get("message", "unknown error") if isinstance(err, dict) else err
            print(f"\n    FAILED: {msg}", flush=True)

    # Streaming: per-delta elapsed_ms. Non-streaming: usage.generation_ms.
    decode_ms = last_elapsed_ms if last_elapsed_ms > 0 else generation_ms
    decode_tokens = last_tokens_gen if last_tokens_gen > 0 else output_tokens

    engine_tok_s = (
        decode_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0
    )
    prefill_tok_s = (
        input_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    )

    return {
        "engine_tok_s": round(engine_tok_s, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "decode_s": round(decode_ms / 1000, 3),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "prefill_ms": round(prefill_ms, 1),
        "ttft_ms": round(ttft_ms, 1),
        "model_info": model_info,
    }
