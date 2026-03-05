"""
Training module for LoRA fine-tuning and from-scratch training.

Provides session-based APIs for:
- LoRA fine-tuning on pre-trained models (TrainingSession)
- From-scratch transformer training (FullTrainingSession)

Quick Start (LoRA)
------------------

    >>> from talu.train import TrainingSession, TrainingConfig, LoraConfig
    >>>
    >>> with TrainingSession() as session:
    ...     session.load_model("./model", lora=LoraConfig(rank=8))
    ...     session.configure(TrainingConfig(total_steps=500))
    ...     session.load_data("./tokens.bin")
    ...     session.run(callback=lambda m: print(f"step {m.step}: loss={m.loss:.4f}"))

Quick Start (From-Scratch)
--------------------------

    >>> from talu.train import FullTrainingSession, TransformerConfig, FullSessionConfig
    >>>
    >>> with FullTrainingSession() as session:
    ...     session.init_model(TransformerConfig(
    ...         vocab_size=32000, d_model=256, num_layers=4, num_heads=4,
    ...     ))
    ...     session.configure(FullSessionConfig(total_steps=10000))
    ...     session.load_data("./tokens.bin")
    ...     session.run(callback=lambda m: print(f"step {m.step}: loss={m.loss:.4f}"))
"""

from ._bindings import LoraConfig, StepMetrics, TrainingConfig, TrainingInfo, TrainingState
from ._full_bindings import (
    FullSessionConfig,
    FullSessionInfo,
    FullSessionState,
    TransformerConfig,
)
from .full_session import FullTrainingSession
from .session import TrainingSession

__all__ = [
    # LoRA fine-tuning
    "TrainingSession",
    "TrainingConfig",
    "LoraConfig",
    "StepMetrics",
    "TrainingInfo",
    "TrainingState",
    # From-scratch training
    "FullTrainingSession",
    "FullSessionConfig",
    "FullSessionInfo",
    "FullSessionState",
    "TransformerConfig",
]
