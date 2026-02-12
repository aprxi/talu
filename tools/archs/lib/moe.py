"""
Mixture of Experts (MoE) implementation.

Provides TopK routing and expert layers with optional MXFP4 quantization.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profiling import ProfileBlock

# Import talu.ops for native operations (optional, only needed at runtime)
try:
    from talu import ops as tk_ops

    def _to_torch(ops_tensor):
        """Convert OpsTensor to torch.Tensor via DLPack (zero-copy)."""
        return torch.from_dlpack(ops_tensor)
except ImportError:
    tk_ops = None
    _to_torch = None


class TopKRouter(nn.Module):
    """Top-K expert routing for MoE layers.

    Routes each token to the top-k experts based on learned routing weights.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        with ProfileBlock("router"):
            router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class Experts(nn.Module):
    """MoE expert layers with optional MXFP4 quantization.

    Supports multiple execution paths:
    - Dense (standard PyTorch) for training/debugging
    - MXFP4 blocked for efficient inference
    - Fused MXFP4 for maximum throughput
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # MXFP4 quantized buffers
        self.register_buffer(
            "gate_up_proj_blocks", torch.empty(self.num_experts, 2 * self.expert_dim, 90, 16, dtype=torch.uint8)
        )
        self.register_buffer(
            "gate_up_proj_scales", torch.empty(self.num_experts, 2 * self.expert_dim, 90, dtype=torch.uint8)
        )
        self.register_buffer(
            "down_proj_blocks", torch.empty(self.num_experts, self.hidden_size, 90, 16, dtype=torch.uint8)
        )
        self.register_buffer(
            "down_proj_scales", torch.empty(self.num_experts, self.hidden_size, 90, dtype=torch.uint8)
        )

        # Dense weights (used when not quantized)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim, dtype=torch.bfloat16)
        )
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size, dtype=torch.bfloat16)
        )
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, dtype=torch.bfloat16))

        # Activation parameters
        self.alpha = 1.702
        self.limit = 7.0

        # Runtime state
        self._use_blocked_weights = False
        self._gate_up_bias_f32 = None
        self._down_bias_f32 = None

    def post_load_hook(self):
        """Called after weights are loaded to finalize quantization state."""
        if self.gate_up_proj_blocks is not None and self.gate_up_proj_blocks.numel() > 0:
            self._use_blocked_weights = True
            if self.gate_up_proj is not None:
                del self.gate_up_proj
                self.gate_up_proj = None
            if self.down_proj is not None:
                del self.down_proj
                self.down_proj = None
            return True
        return False

    def _ensure_cached_buffers(self, device: torch.device) -> None:
        if self._gate_up_bias_f32 is None or self._gate_up_bias_f32.device != device:
            self._gate_up_bias_f32 = self.gate_up_proj_bias.float().contiguous()
        if self._down_bias_f32 is None or self._down_bias_f32.device != device:
            self._down_bias_f32 = self.down_proj_bias.float().contiguous()

    def _experts_small(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Small batch path: process each expert separately using talu MXFP4."""
        next_states = torch.zeros(hidden_states.shape, device=hidden_states.device, dtype=torch.float32)
        expert_mask = F.one_hot(router_indices.long(), num_classes=num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            idx = int(expert_idx[0].item())
            if idx == num_experts:
                continue

            _, token_idx = torch.where(expert_mask[idx])
            current_state = hidden_states[token_idx]

            # MXFP4 gate_up projection
            gate_up = _to_torch(tk_ops.mxfp4_matmul(
                current_state,
                self.gate_up_proj_blocks[idx],
                self.gate_up_proj_scales[idx],
                self._gate_up_bias_f32[idx],
            ))

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu

            # MXFP4 down projection
            out = _to_torch(tk_ops.mxfp4_matmul(
                gated_output,
                self.down_proj_blocks[idx],
                self.down_proj_scales[idx],
                self._down_bias_f32[idx],
            ))

            weighted_output = out * routing_weights[token_idx, idx, None]
            next_states.index_add_(0, token_idx, weighted_output)

        return next_states

    def _experts_dense(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Dense path: standard PyTorch for training/debugging."""
        next_states = torch.zeros(hidden_states.shape, device=hidden_states.device, dtype=torch.float32)
        expert_mask = F.one_hot(router_indices, num_classes=num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            idx = int(expert_idx[0].item())
            if idx == num_experts:
                continue

            _, token_idx = torch.where(expert_mask[idx])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[idx] + self.gate_up_proj_bias[idx]

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu

            out = gated_output @ self.down_proj[idx] + self.down_proj_bias[idx]
            weighted_output = out * routing_weights[token_idx, idx, None]
            next_states.index_add_(0, token_idx, weighted_output)

        return next_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        input_dtype = hidden_states.dtype
        num_experts = routing_weights.shape[1]

        with ProfileBlock("experts"):
            if self._use_blocked_weights:
                router_indices = router_indices.to(torch.int32)
                self._ensure_cached_buffers(hidden_states.device)
                next_states = self._experts_small(hidden_states, router_indices, routing_weights, num_experts)
            else:
                next_states = self._experts_dense(hidden_states, router_indices, routing_weights, num_experts)

        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states.to(input_dtype)


class MoEMLP(nn.Module):
    """Complete MoE MLP layer combining router and experts."""

    def __init__(self, config):
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = Experts(config)

    def post_load_hook(self):
        return self.experts.post_load_hook()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores
