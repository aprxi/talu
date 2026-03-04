"""
Training module for LoRA fine-tuning.

Provides a session-based API for training LoRA adapters on top of
pre-trained language models.

Quick Start
-----------

    >>> from talu.train import TrainingSession, TrainingConfig, LoraConfig
    >>>
    >>> with TrainingSession() as session:
    ...     session.load_model("./model", lora=LoraConfig(rank=8))
    ...     session.configure(TrainingConfig(total_steps=500))
    ...     session.load_data("./tokens.bin")
    ...     session.run(callback=lambda m: print(f"step {m.step}: loss={m.loss:.4f}"))
"""

from ._bindings import LoraConfig, StepMetrics, TrainingConfig, TrainingInfo, TrainingState
from .session import TrainingSession

__all__ = [
    "TrainingSession",
    "TrainingConfig",
    "LoraConfig",
    "StepMetrics",
    "TrainingInfo",
    "TrainingState",
]
