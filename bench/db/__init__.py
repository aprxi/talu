"""Auto-import all DB benchmark scenario modules from subdirectories."""

from pathlib import Path
import importlib

_pkg = Path(__file__).parent
for f in sorted(_pkg.rglob("*.py")):
    if f.name.startswith("_"):
        continue
    rel = f.relative_to(_pkg).with_suffix("")
    module = ".".join(rel.parts)
    importlib.import_module(f".{module}", __package__)
