"""
Search - Find models on a hub with filtering and limits.

Primary API: talu.Repository
Scope: Single

Use `search()` to discover text-generation models on a hub.
Results are filtered to text-generation models by default.

Related:
- examples/basics/14_model_repository.py
"""

from talu.repository import Repository

repo = Repository()

# =============================================================================
# Basic search
# =============================================================================

print("Search for 'qwen':")
for model_id in repo.search("qwen"):
    print(f"  {model_id}")

# =============================================================================
# Search with limit
# =============================================================================

print("\nTop 5 results for 'llama':")
for model_id in repo.search("llama", limit=5):
    print(f"  {model_id}")

# =============================================================================
# Check which results are cached
# =============================================================================

print("\nSearch 'phi' (showing cache status):")
for model_id in repo.search("phi", limit=10):
    if repo.is_cached(model_id):
        print(f"  {model_id}  [cached]")
    else:
        print(f"  {model_id}")

# =============================================================================
# Search and fetch workflow
# =============================================================================


def search_and_preview(query: str, limit: int = 5):
    """Search for models and show their details."""
    print(f"\nSearching for '{query}':")
    results = list(repo.search(query, limit=limit))

    if not results:
        print("  No results found")
        return

    for model_id in results:
        cached = repo.is_cached(model_id)
        status = "[cached]" if cached else ""

        if cached:
            size = repo.size(model_id)
            print(f"  {model_id}  {status}  ({size / 1e6:.1f} MB)")
        else:
            print(f"  {model_id}  {status}")


search_and_preview("mistral")
search_and_preview("gemma")

# =============================================================================
# Search with authentication (for private models)
# =============================================================================

import os

token = os.environ.get("HF_TOKEN")

if token:
    print("\nSearching with authentication:")
    for model_id in repo.search("private-model", token=token, limit=5):
        print(f"  {model_id}")

# =============================================================================
# Interactive model selection
# =============================================================================


def select_and_fetch(query: str):
    """Search, display options, and fetch selected model."""
    results = list(repo.search(query, limit=10))

    if not results:
        print(f"No models found for '{query}'")
        return None

    print(f"\nFound {len(results)} models for '{query}':")
    for i, model_id in enumerate(results, 1):
        cached = "[cached]" if repo.is_cached(model_id) else ""
        print(f"  {i}. {model_id} {cached}")

    # In a real app, you'd get user input here
    # choice = int(input("Select model (1-10): ")) - 1
    choice = 0  # Default to first result for example

    selected = results[choice]
    print(f"\nSelected: {selected}")

    if not repo.is_cached(selected):
        print("Downloading...")
        path = repo.fetch(selected)
        print(f"Downloaded to: {path}")
    else:
        path = repo.cache_path(selected)
        print(f"Using cached: {path}")

    return path


# Uncomment to run interactive selection:
# path = select_and_fetch("qwen 0.6b")

"""
Topics covered:
* repository.search
* search.scoring
"""
