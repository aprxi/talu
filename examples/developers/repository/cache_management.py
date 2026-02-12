"""
Cache Management - Monitor disk usage and find largest models.

Primary API: talu.Repository
Scope: Single

The Repository API provides tools for managing cache size and cleaning up
unused models.

Related:
- examples/basics/14_model_repository.py
"""

from talu.repository import Repository

repo = Repository()


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# =============================================================================
# Check cache size
# =============================================================================

# Total cache size
total_size = repo.size()
print(f"Total cache size: {format_size(total_size)}")

# Size of specific model
model = "Qwen/Qwen3-0.6B"
if repo.is_cached(model):
    model_size = repo.size(model)
    print(f"Size of {model}: {format_size(model_size)}")

# =============================================================================
# Disk usage summary (like `du`)
# =============================================================================

print("\nCache usage by model:")
models = list(repo.list_models())
for model_id in sorted(models):
    size = repo.size(model_id)
    print(f"  {format_size(size)}\t{model_id}")

print(f"  {format_size(total_size)}\ttotal")

# =============================================================================
# Find largest models
# =============================================================================

print("\nLargest models:")
model_sizes = [(model_id, repo.size(model_id)) for model_id in repo.list_models()]
model_sizes.sort(key=lambda x: x[1], reverse=True)

for model_id, size in model_sizes[:5]:
    print(f"  {format_size(size)}\t{model_id}")

# =============================================================================
# Delete a model
# =============================================================================

model_to_delete = "some-model/to-delete"

if repo.is_cached(model_to_delete):
    size = repo.size(model_to_delete)
    print(f"\nDeleting {model_to_delete} ({format_size(size)})...")

    if repo.delete(model_to_delete):
        print("Deleted successfully")
    else:
        print("Delete failed")

# =============================================================================
# Clear all models (use with caution!)
# =============================================================================

# Uncomment to clear entire cache:
# count = repo.clear()
# print(f"Deleted {count} models")

# =============================================================================
# Cleanup: remove models above size threshold
# =============================================================================


def cleanup_large_models(threshold_gb: float = 10.0, dry_run: bool = True):
    """Remove models larger than threshold."""
    threshold_bytes = int(threshold_gb * 1024 * 1024 * 1024)
    removed = 0
    freed = 0

    for model_id in list(repo.list_models()):
        size = repo.size(model_id)
        if size > threshold_bytes:
            if dry_run:
                print(f"  Would delete: {model_id} ({format_size(size)})")
            else:
                if repo.delete(model_id):
                    print(f"  Deleted: {model_id} ({format_size(size)})")
                    removed += 1
                    freed += size

    if dry_run:
        print("(dry run - no models deleted)")
    else:
        print(f"Deleted {removed} models, freed {format_size(freed)}")


print("\nModels larger than 10 GB:")
cleanup_large_models(threshold_gb=10.0, dry_run=True)

# =============================================================================
# Check cache directory
# =============================================================================

print(f"\nCache directory: {repo.cache_dir}")

# Cache path layout example:
# <cache_dir>/models--org--name/snapshots/<revision>/

"""
Topics covered:
* repository.cache
* repository.inspect
"""
