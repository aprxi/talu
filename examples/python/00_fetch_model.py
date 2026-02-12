"""Fetch a model before first use.

Run this before other examples to download the model.

Usage:
    python examples/python/00_fetch_model.py
    python examples/python/00_fetch_model.py --force-update

    # Optionally use a different model:
    MODEL_URI=Qwen/Qwen3-0.6B python examples/python/00_fetch_model.py
"""

import os
import sys

from talu import repository

if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__.strip())
    sys.exit(0)

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")
force = "--force-update" in sys.argv

if not force and repository.is_cached(MODEL_URI):
    path = repository.cache_path(MODEL_URI)
    print(f"Model '{MODEL_URI}' is ready at: {path}")
    sys.exit(0)

if force:
    prompt = f"Re-download '{MODEL_URI}' from HuggingFace? [y/N] "
else:
    prompt = f"Model '{MODEL_URI}' not found locally. Download from HuggingFace? [y/N] "

answer = input(prompt)
if answer.strip().lower() != "y":
    print("Aborted.")
    sys.exit(1)

_last_file = ""


def show_progress(downloaded: int, total: int, filename: str) -> None:
    global _last_file
    if filename != _last_file:
        if _last_file:
            print()
        _last_file = filename
    if total > 0:
        pct = downloaded * 100 // total
        mb_done = downloaded / 1e6
        mb_total = total / 1e6
        print(f"\r  {filename}: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
    else:
        print(f"\r  {filename}: {downloaded / 1e6:.1f} MB", end="", flush=True)


path = repository.fetch(MODEL_URI, force=force, on_progress=show_progress)
print()

if path:
    print(f"Model '{MODEL_URI}' is ready at: {path}")
else:
    print("Download failed.", file=sys.stderr)
    sys.exit(1)
