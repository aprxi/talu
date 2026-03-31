"""MMMU evaluation scenario for POST /v1/responses.

Registers responses/evals/mmmu.  Measures multimodal reasoning accuracy
across diverse academic subjects requiring image understanding.

Dataset: MMMU/MMMU, validation split.
"""

from __future__ import annotations

import hashlib
import json
import socket
import tempfile
import urllib.parse

from scenario import Scenario
from responses.evals._runner import run_eval

_INSTRUCTIONS = (
    "You are taking a multimodal multiple choice exam. "
    "Look at the provided image(s) and read the question carefully. "
    "Respond with ONLY the letter of the correct answer. "
    "Do not explain your reasoning."
)

_API_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


def _upload_file(base_url: str, filepath: str, filename: str, content_type: str = "image/png") -> str:
    """Upload a file via POST /v1/files (multipart/form-data). Returns file_id."""
    parsed = urllib.parse.urlparse(base_url)
    addr = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    with open(filepath, "rb") as f:
        file_data = f.read()

    boundary = "----talu-eval-upload"
    body_parts = [
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n"
        f"\r\n",
    ]
    body = body_parts[0].encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((addr, port))

    request = (
        f"POST /v1/files HTTP/1.1\r\n"
        f"Host: {addr}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body

    sock.sendall(request)

    chunks: list[bytes] = []
    while True:
        try:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break
    sock.close()

    raw = b"".join(chunks).decode(errors="replace")
    if "\r\n\r\n" in raw:
        _, body_str = raw.split("\r\n\r\n", 1)
    else:
        body_str = raw

    resp = json.loads(body_str.strip())
    file_id = resp.get("id", "")
    if not file_id:
        raise RuntimeError(f"File upload failed: {resp}")
    return file_id


def _save_pil_image(img) -> str:
    """Save a PIL image to a temporary PNG file. Returns the file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    return tmp.name


def _load_dataset(n: int | None = None):
    """Load MMMU validation samples."""
    from datasets import load_dataset

    try:
        ds = load_dataset("MMMU/MMMU", split="validation")
    except Exception as exc:
        if "gated" in str(exc).lower():
            raise SystemExit(
                "MMMU is a gated dataset. Accept access at:\n"
                "  https://huggingface.co/datasets/MMMU/MMMU\n"
                "Then ensure your HF token is available (HF_HOME or huggingface-cli login)."
            ) from exc
        raise
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        options = row["options"]
        if isinstance(options, str):
            options = json.loads(options)

        choices_str = "\n".join(
            f"{chr(65 + j)}. {c}" for j, c in enumerate(options)
        )
        prompt = f"{row['question']}\n\n{choices_str}"

        # Collect images (up to 7 per question).
        images = []
        for k in range(1, 8):
            img = row.get(f"image_{k}")
            if img is not None:
                images.append(img)

        samples.append({
            "prompt": prompt,
            "correct": row["answer"],
            "question_hash": hashlib.sha256(row["question"].encode()).hexdigest()[:16],
            "index": i,
            "images": images,
        })
    return samples


def _make_build_body(base_url: str):
    """Return a build_body closure that captures base_url for image upload."""

    def _build_body(sample: dict, uri: str, config: dict) -> dict:
        # Upload images and build multimodal content.
        content: list[dict] = [
            {"type": "input_text", "text": sample["prompt"]},
        ]
        for img in sample.get("images", []):
            tmp_path = _save_pil_image(img)
            file_id = _upload_file(base_url, tmp_path, "image.png")
            content.append({"type": "input_image", "image_url": file_id})

        body: dict = {
            "model": uri,
            "input": [{
                "type": "message",
                "role": "user",
                "content": content,
            }],
            "instructions": _INSTRUCTIONS,
            "stream": False,
            "store": False,
        }
        if "max_tokens" in config:
            body["max_output_tokens"] = config["max_tokens"]
        for cfg_key, api_key in _API_FIELDS.items():
            if cfg_key in config:
                body[api_key] = config[cfg_key]
        mrt = int(config.get("max_reasoning_tokens", 0))
        body["max_reasoning_tokens"] = mrt
        return body

    return _build_body


class Mmmu(Scenario):
    name = "responses/evals/mmmu"
    description = "MMMU — multimodal reasoning accuracy."
    endpoint = "POST /v1/responses"

    def prepare_config(self, config: dict) -> None:
        # Default to non-thinking mode (user can override with --set max_reasoning_tokens=N).
        if "max_reasoning_tokens" not in config:
            config["max_reasoning_tokens"] = 0

    def run(self, base_url: str, rounds: int, config: dict) -> list[dict]:
        samples_n: int | None = config.get("samples")
        if isinstance(samples_n, str):
            samples_n = int(samples_n)

        print("  Loading MMMU dataset ...", flush=True)
        samples = _load_dataset(samples_n)
        print(f"  {len(samples)} samples loaded.\n", flush=True)

        return run_eval(
            bench_name="mmmu",
            base_url=base_url,
            config=config,
            samples=samples,
            build_body=_make_build_body(base_url),
        )
