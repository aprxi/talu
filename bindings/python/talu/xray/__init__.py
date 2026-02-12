"""
X-Ray - Tensor Inspection API.

Capture and analyze tensor values during inference for debugging
numerical issues and validating implementations.

Example:
    >>> from talu.xray import Inspector, CaptureMode
    >>>
    >>> # Create inspector for lm_head and block outputs
    >>> inspector = Inspector(
    ...     points=["lm_head", "block_out"],
    ...     mode=CaptureMode.STATS,
    ... )
    >>>
    >>> # Use as context manager
    >>> with inspector:
    ...     # Run inference...
    ...     pass
    >>>
    >>> # Query results
    >>> print(f"Captured {len(inspector)} tensors")
    >>> for tensor in inspector:
    ...     print(f"{tensor.point_name} L{tensor.layer}: "
    ...           f"min={tensor.stats.min:.4f} max={tensor.stats.max:.4f}")
    >>>
    >>> # Check for anomalies
    >>> if anomaly := inspector.find_anomaly():
    ...     print(f"Found NaN at {anomaly.point_name}")
"""

from ._bindings import (
    # Bitmask constants
    POINT_ALL,
    POINT_BLOCK_OUT,
    POINT_CONV_CONV,
    POINT_CONV_IN_PROJ,
    POINT_CONV_OUT_PROJ,
    POINT_EMBED,
    POINT_FINAL_NORM,
    POINT_LM_HEAD,
    POINT_LOGITS_SCALED,
    POINT_MAMBA_OUT,
    AnomalyLocation,
    CapturedTensor,
    CaptureMode,
    Point,
    Stats,
)
from .inspector import Inspector

__all__ = [
    # Main class
    "Inspector",
    # Data types
    "CapturedTensor",
    "Stats",
    "AnomalyLocation",
    # Enums
    "CaptureMode",
    "Point",
    # Constants
    "POINT_ALL",
    "POINT_EMBED",
    "POINT_BLOCK_OUT",
    "POINT_MAMBA_OUT",
    "POINT_CONV_IN_PROJ",
    "POINT_CONV_CONV",
    "POINT_CONV_OUT_PROJ",
    "POINT_FINAL_NORM",
    "POINT_LM_HEAD",
    "POINT_LOGITS_SCALED",
]
