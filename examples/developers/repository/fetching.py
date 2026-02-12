"""
Fetching - Download models with progress callbacks and force re-download.

Primary API: talu.Repository
Scope: Single

Use `fetch()` to download models to the local cache. Models are stored in the
standard cache layout under the configured cache directory.

Related:
- examples/basics/14_model_repository.py
"""

from talu.repository import Repository

repo = Repository()

# =============================================================================
# Basic fetch
# =============================================================================

model = "Qwen/Qwen3-0.6B"

# Download if not cached, returns path to model directory
path = repo.fetch(model)
print(f"Model path: {path}")

# =============================================================================
# Fetch with progress callback
# =============================================================================


def on_progress(downloaded: int, total: int, filename: str):
    """Progress callback for download progress."""
    if total > 0:
        pct = (downloaded * 100) // total
        print(f"\r  {filename}: {downloaded // 1024}KB / {total // 1024}KB ({pct}%)", end="")
    else:
        print(f"\r  {filename}: {downloaded // 1024}KB", end="")


print(f"\nDownloading {model} with progress:")
path = repo.fetch(model, on_progress=on_progress)
print(f"\nDone: {path}")

# =============================================================================
# Force re-download
# =============================================================================

# Re-download even if already cached
path = repo.fetch(model, force=True, on_progress=on_progress)
print(f"\nRe-downloaded: {path}")

# =============================================================================
# Private models with token
# =============================================================================

# Set HF_TOKEN environment variable, or pass token directly
import os

token = os.environ.get("HF_TOKEN")

if token:
    # Download private model
    path = repo.fetch("your-org/private-model", token=token)
    print(f"Private model: {path}")

# =============================================================================
# Check cache before fetch
# =============================================================================

model = "meta-llama/Llama-3.2-1B"

# Check if already cached (no network call)
if repo.is_cached(model):
    print(f"\n{model} already cached, using local copy")
    path = repo.cache_path(model)
else:
    print(f"\n{model} not cached, downloading...")
    path = repo.fetch(model)

if path:
    print(f"  Path: {path}")
else:
    print(f"  Failed to get {model}")

# =============================================================================
# Batch download
# =============================================================================

models_to_fetch = [
    "Qwen/Qwen3-0.6B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

print("\nBatch download:")
for model_id in models_to_fetch:
    if repo.is_cached(model_id):
        print(f"  {model_id}: already cached")
    else:
        print(f"  {model_id}: downloading...")
        repo.fetch(model_id)
        print(f"  {model_id}: done")

"""
Topics covered:
* repository.fetch
* download.progress
"""
