"""Manage models — cache, download, and inspect files.

This example shows:
- Listing cached models
- Downloading models from the hub
- Inspecting model files and sizes
- Checking cache size and searching for models
"""

import os
import sys

from talu import repository

MODEL_URI = os.environ.get("MODEL_URI", "LiquidAI/LFM2-350M")

# See where models are cached
print(f"Cache directory: {repository.cache_dir()}")

# List cached models
print("\nCached models:")
cached = list(repository.list_models())
if cached:
    for model_id in cached:
        print(f"  {model_id}")
else:
    print("  (none)")

model = MODEL_URI

# Check local cache only (fast, no network)
if repository.is_cached(model):
    print(f"\n{model} is cached locally")
    path = repository.cache_path(model)
    print(f"  Path: {path}")
else:
    answer = input(f"\n{model} not cached. Download from HuggingFace? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)
    path = repository.fetch(
        model,
        on_progress=lambda done, total, name: print(
            f"\r  {name}: {done * 100 // total}%" if total > 0 else f"\r  {name}: {done / 1e6:.1f} MB",
            end="", flush=True,
        ),
    )
    print()
    print(f"Downloaded {model} to: {path}")

# List files if we have a local copy
if path:
    files = list(repository.list_files(path))
    print(f"\nFiles in {model}: {len(files)}")
    print(f"First 5 files: {files[:5]}")
    size = repository.size(model)
    print(f"Model size: {size / 1e6:.1f} MB")

# List files from hub (checks cache first, then fetches file list from hub)
try:
    remote_files = list(repository.list_files(model))
    print(f"\nFiles available: {len(remote_files)}")
except Exception as exc:
    print(f"File listing failed: {exc}")

# Total cache size
total = repository.size()
print(f"\nTotal cache size: {total / 1e9:.2f} GB")

# Search for models (requires network)
try:
    results = list(repository.search("qwen 0.6b", limit=5))
    if results:
        print(f"\nSearch results: {len(results)} matches")
        for result in results[:3]:  # Show first 3
            print(f"  - {result}")
    else:
        print("\nNo search results found")
except Exception as exc:
    print(f"\nSearch not available: {exc}")

# Optional: delete a cached model (disabled by default)
# if repository.delete(model):
#     print(f"Deleted {model}")
#     print(f"New cache size: {repository.size() / 1e9:.2f} GB")

