"""
Lightweight Module system for inference.

Provides a minimal nn.Module replacement that uses talu.ops for computation.
Uses numpy arrays for weight storage (talu supports numpy via DLPack).

This eliminates:
- nn.Module inheritance and overhead
- nn.Parameter gradient tracking
- Training-related features (optimizer state, hooks, etc.)
- torch dependency (except when nn.Module children are used)
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, TYPE_CHECKING

import numpy as np
from safetensors import safe_open
from talu import ops

if TYPE_CHECKING:
    import torch


class Parameter:
    """A tensor that should be loaded from weights. No gradient tracking."""

    def __init__(self, shape: Tuple[int, ...] = None, dtype: np.dtype = np.float32):
        self.shape = shape
        self.dtype = dtype
        self.data: Optional[np.ndarray] = None

    def set(self, data: np.ndarray):
        """Set the parameter data (no clone, just reference)."""
        self.data = data

    def get(self) -> np.ndarray:
        """Get the parameter data for computation."""
        if self.data is None:
            raise ValueError("Parameter not initialized")
        return self.data

    def __repr__(self):
        if self.data is not None:
            return f"Parameter(shape={tuple(self.data.shape)}, dtype={self.data.dtype})"
        return f"Parameter(uninitialized)"


class Module:
    """
    Base class for torch-free modules.

    Similar interface to nn.Module but without the torch dependency.
    Uses talu.ops for computation.

    Also supports nn.Module children for gradual migration.
    """

    def __init__(self):
        self._modules: Dict[str, "Module"] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self._nn_modules: Dict[str, Any] = {}  # For nn.Module children (requires torch)

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if "_parameters" not in self.__dict__:
                object.__setattr__(self, "_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if "_modules" not in self.__dict__:
                object.__setattr__(self, "_modules", {})
            self._modules[name] = value
        elif isinstance(value, ModuleList):
            if "_modules" not in self.__dict__:
                object.__setattr__(self, "_modules", {})
            # Store ModuleList items with indexed names
            for i, mod in enumerate(value):
                self._modules[f"{name}.{i}"] = mod
            object.__setattr__(self, name, value)
        elif hasattr(value, '__class__') and hasattr(value.__class__, '__mro__'):
            # Check if it's an nn.Module (has named_parameters method)
            if hasattr(value, 'named_parameters') and hasattr(value, 'named_modules'):
                if "_nn_modules" not in self.__dict__:
                    object.__setattr__(self, "_nn_modules", {})
                self._nn_modules[name] = value
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """Iterate over all parameters with their full names."""
        # Own parameters
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        # Lightweight module children
        for mod_name, module in self._modules.items():
            mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(mod_prefix)
        # nn.Module children (wrap their parameters)
        for mod_name, nn_module in self._nn_modules.items():
            mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            for param_name, param in nn_module.named_parameters():
                full_name = f"{mod_prefix}.{param_name}"
                # Wrap nn.Parameter in our Parameter
                wrapper = Parameter()
                wrapper.data = param.data
                yield full_name, wrapper

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        """Iterate over all modules with their full names."""
        yield prefix, self
        # Lightweight module children
        for name, module in self._modules.items():
            mod_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(mod_prefix)
        # nn.Module children
        for name, nn_module in self._nn_modules.items():
            mod_prefix = f"{prefix}.{name}" if prefix else name
            for sub_name, sub_module in nn_module.named_modules():
                full_name = f"{mod_prefix}.{sub_name}" if sub_name else mod_prefix
                yield full_name, sub_module

    def register_buffer(self, name: str, data: np.ndarray):
        """Register a non-parameter buffer."""
        self._buffers[name] = data
        object.__setattr__(self, name, data)


class ModuleList(list):
    """List container for modules."""

    def __init__(self, modules: list = None):
        super().__init__(modules or [])

    def __setitem__(self, idx: int, module: Module):
        super().__setitem__(idx, module)

    def append(self, module: Module):
        super().append(module)


class Linear(Module):
    """Linear layer using talu.ops.linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features))
        if bias:
            self.bias = Parameter((out_features,))
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias.get() if self.bias else None
        return ops.linear(x, self.weight.get(), bias)


class Embedding(Module):
    """Embedding layer using talu.ops.embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter((num_embeddings, embedding_dim))

    def forward(self, indices):
        return ops.embedding(indices=indices, weight=self.weight.get())


class RMSNorm(Module):
    """RMSNorm using talu.ops.rms_norm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter((hidden_size,))
        self.variance_epsilon = eps

    def forward(self, x):
        return ops.rms_norm(x, self.weight.get(), self.variance_epsilon)


# === Weight Loading ===

def find_safetensor_files(model_path: str) -> Tuple[Path, list]:
    """Find safetensor files in a model directory."""
    path = Path(model_path)

    if path.is_file() and path.suffix == ".safetensors":
        return path.parent, [path.name]

    if (path / "model.safetensors").exists():
        return path, ["model.safetensors"]

    if (path / "model.safetensors.index.json").exists():
        with open(path / "model.safetensors.index.json") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return path, files

    # Try HuggingFace cache structure
    snapshots = path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshot_dirs:
            return find_safetensor_files(str(snapshot_dirs[0]))

    raise FileNotFoundError(f"No safetensor files found in {model_path}")


def load_safetensors(path: str, use_torch: bool = None):
    """
    Load safetensors file(s).

    Handles both single files and sharded models.

    Args:
        path: Path to model directory or safetensors file
        use_torch: If True, load as torch tensors. If False, load as numpy.
                   If None (default), auto-detect based on dtype (bf16 needs torch).

    Returns:
        Dict mapping tensor names to tensors (torch.Tensor or np.ndarray)
    """
    base_path, files = find_safetensor_files(path)
    weights = {}

    for filepath in files:
        full_path = str(base_path / filepath)

        if use_torch is None:
            # Auto-detect: try numpy first, fall back to torch for bf16
            try:
                with safe_open(full_path, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            except TypeError as e:
                if "bfloat16" in str(e):
                    # bf16 requires torch - load as torch tensors (no conversion)
                    with safe_open(full_path, framework="pt") as f:
                        for key in f.keys():
                            weights[key] = f.get_tensor(key)
                else:
                    raise
        elif use_torch:
            with safe_open(full_path, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            with safe_open(full_path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

    return weights


def load_weights_into_model(model: Module, weights: Dict[str, Any], strict: bool = False):
    """
    Load weights into a Module by matching parameter names.

    Handles both lightweight Module parameters and nn.Module parameters.
    Accepts both numpy arrays and torch tensors as weights.

    Args:
        model: The model to load weights into
        weights: Dict of name -> tensor (np.ndarray or torch.Tensor)
        strict: If True, raise error for missing/extra weights
    """
    loaded = set()
    missing = []
    _torch = None  # Lazy import for nn.Module children

    def _get_torch():
        nonlocal _torch
        if _torch is None:
            import torch as _t
            _torch = _t
        return _torch

    def _to_torch(tensor):
        """Convert tensor to torch if needed."""
        torch = _get_torch()
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor)
        return tensor  # Already torch.Tensor

    # Load into lightweight Module parameters
    def load_params(mod, prefix=""):
        for name, param in mod._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            candidates = [full_name, f"model.{full_name}"]
            for candidate in candidates:
                if candidate in weights:
                    param.set(weights[candidate])
                    loaded.add(candidate)
                    break
            else:
                missing.append(full_name)

        for mod_name, child in mod._modules.items():
            child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            load_params(child, child_prefix)

        # Load into nn.Module children (requires torch)
        for mod_name, nn_mod in mod._nn_modules.items():
            torch = _get_torch()
            nn_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            for param_name, param in nn_mod.named_parameters():
                full_name = f"{nn_prefix}.{param_name}"
                candidates = [full_name, f"model.{full_name}"]
                for candidate in candidates:
                    if candidate in weights:
                        # Set param data (inference only, no gradients)
                        param.data = _to_torch(weights[candidate])
                        param.requires_grad_(False)
                        loaded.add(candidate)
                        break
                else:
                    missing.append(full_name)

            # Also load buffers for nn.Module
            for buf_name, buf in nn_mod.named_buffers():
                full_name = f"{nn_prefix}.{buf_name}"
                candidates = [full_name, f"model.{full_name}"]
                for candidate in candidates:
                    if candidate in weights:
                        parts = buf_name.split(".")
                        target = nn_mod
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        setattr(target, parts[-1], _to_torch(weights[candidate]))
                        loaded.add(candidate)
                        break

    load_params(model)

    extra = set(weights.keys()) - loaded

    if strict:
        if missing:
            raise KeyError(f"Missing weights: {missing}")
        if extra:
            raise KeyError(f"Extra weights: {extra}")

    return {"loaded": len(loaded), "missing": len(missing), "extra": len(extra)}


def load_config(model_path: str, config_name: str = "config.json") -> dict:
    """Load model config from a directory or HuggingFace cache path."""
    path = Path(model_path)
    config_path = path / config_name

    if not config_path.exists():
        snapshots = path / "snapshots"
        if snapshots.exists():
            snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshot_dirs:
                config_path = snapshot_dirs[0] / config_name

    with open(config_path) as f:
        return json.load(f)
