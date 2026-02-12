"""Tests for talu.converter.inspector module.

Tests for ModelInfo and describe() function.
Note: Full validation tests are in tests/reference/models/test_describe.py
"""

import pytest

from talu.converter import ModelInfo


class TestModelInfo:
    """Tests for ModelInfo class."""

    def test_model_info_construction(self):
        """ModelInfo can be constructed with all required parameters."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        assert info.vocab_size == 32000
        assert info.hidden_size == 4096
        assert info.num_layers == 32
        assert info.num_heads == 32
        assert info.num_kv_heads == 8
        assert info.intermediate_size == 14336
        assert info.max_seq_len == 4096
        assert info.head_dim == 128
        assert info.rope_theta == 10000.0
        assert info.norm_eps == 1e-5
        assert info.quant_bits == 16
        assert info.quant_group_size == 128
        assert info.model_type == "llama"
        assert info.architecture == "LlamaForCausalLM"
        assert info.tie_word_embeddings is False
        assert info.use_gelu is False
        assert info.num_experts == 0
        assert info.experts_per_token == 0

    def test_is_quantized_fp16(self):
        """is_quantized returns False for 16-bit models."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        assert info.is_quantized is False

    def test_is_quantized_4bit(self):
        """is_quantized returns True for 4-bit models."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=4,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        assert info.is_quantized is True

    def test_is_quantized_8bit(self):
        """is_quantized returns True for 8-bit models."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=8,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        assert info.is_quantized is True

    def test_is_moe_false(self):
        """is_moe returns False for non-MoE models."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        assert info.is_moe is False

    def test_is_moe_true(self):
        """is_moe returns True for MoE models."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type="mixtral",
            architecture="MixtralForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=8,
            experts_per_token=2,
        )

        assert info.is_moe is True

    def test_repr(self):
        """ModelInfo has a useful repr."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        r = repr(info)
        assert "LlamaForCausalLM" in r
        assert "32" in r  # layers
        assert "4096" in r  # hidden
        assert "FP16" in r  # not quantized

    def test_repr_quantized(self):
        """ModelInfo repr shows quantization level."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=4,
            quant_group_size=128,
            model_type="llama",
            architecture="LlamaForCausalLM",
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        r = repr(info)
        assert "Q4" in r

    def test_none_model_type_and_architecture(self):
        """ModelInfo handles None for model_type and architecture."""
        info = ModelInfo(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            max_seq_len=4096,
            head_dim=128,
            rope_theta=10000.0,
            norm_eps=1e-5,
            quant_bits=16,
            quant_group_size=128,
            model_type=None,
            architecture=None,
            tie_word_embeddings=False,
            use_gelu=False,
            num_experts=0,
            experts_per_token=0,
        )

        # Should not raise
        r = repr(info)
        assert "unknown" in r


class TestDescribeFunction:
    """Tests for describe() function.

    Note: Tests requiring actual model files are in tests/reference/models/.
    These tests verify API behavior without requiring model weights.
    """

    def test_describe_raises_on_invalid_path(self):
        """describe() raises ModelError for invalid paths."""
        from talu.converter import describe
        from talu.exceptions import ModelError

        with pytest.raises(ModelError):
            describe("/nonexistent/path/to/model")

    def test_describe_raises_on_invalid_model_id(self):
        """describe() raises ModelError for invalid model IDs."""
        from talu.converter import describe
        from talu.exceptions import ModelError

        with pytest.raises(ModelError):
            describe("definitely-not-a-real-org/not-a-real-model-12345")


class TestModelInfoCStructLayout:
    """Tests that Python ModelInfoC matches Zig ModelInfo struct layout.

    These tests prevent ABI mismatches that cause segfaults.
    If any test fails, the Python struct definition is out of sync
    with capi/converter.zig ModelInfo.
    """

    def test_struct_has_expected_fields(self):
        """ModelInfoC has all expected fields in correct order."""
        import ctypes

        from talu._native import ModelInfo as ModelInfoC

        # Expected fields in order (must match Zig struct exactly)
        expected_fields = [
            ("vocab_size", ctypes.c_int32),
            ("hidden_size", ctypes.c_int32),
            ("num_layers", ctypes.c_int32),
            ("num_heads", ctypes.c_int32),
            ("num_kv_heads", ctypes.c_int32),
            ("intermediate_size", ctypes.c_int32),
            ("max_seq_len", ctypes.c_int32),
            ("head_dim", ctypes.c_int32),
            ("rope_theta", ctypes.c_float),
            ("norm_eps", ctypes.c_float),
            ("quant_bits", ctypes.c_int32),
            ("quant_group_size", ctypes.c_int32),
            ("quant_method", ctypes.c_int32),
            ("model_type", ctypes.c_void_p),
            ("architecture", ctypes.c_void_p),
            ("tie_word_embeddings", ctypes.c_bool),
            ("use_gelu", ctypes.c_bool),
            ("num_experts", ctypes.c_int32),
            ("experts_per_token", ctypes.c_int32),
            ("error_msg", ctypes.c_char_p),
        ]

        actual_fields = ModelInfoC._fields_

        assert len(actual_fields) == len(expected_fields), (
            f"Field count mismatch: expected {len(expected_fields)}, got {len(actual_fields)}"
        )

        for i, ((exp_name, exp_type), (act_name, act_type)) in enumerate(
            zip(expected_fields, actual_fields, strict=True)
        ):
            assert exp_name == act_name, (
                f"Field {i} name mismatch: expected '{exp_name}', got '{act_name}'"
            )
            assert exp_type == act_type, (
                f"Field '{exp_name}' type mismatch: expected {exp_type}, got {act_type}"
            )

    def test_error_msg_is_last_field(self):
        """error_msg must be the last field for error checking to work."""
        from talu._native import ModelInfo as ModelInfoC

        fields = ModelInfoC._fields_
        last_field_name, _ = fields[-1]
        assert last_field_name == "error_msg", (
            f"Last field must be 'error_msg', got '{last_field_name}'"
        )

    def test_struct_size_is_reasonable(self):
        """ModelInfoC size should be reasonable for the field count.

        This catches gross misalignments. The struct has:
        - 12 x i32 (48 bytes)
        - 2 x f32 (8 bytes)
        - 2 x pointer (16 bytes on 64-bit)
        - 2 x bool (2 bytes, but may be padded)
        - 1 x char_p (8 bytes on 64-bit)
        Total: ~82+ bytes, with alignment padding typically 88-96 bytes
        """
        import ctypes

        from talu._native import ModelInfo as ModelInfoC

        size = ctypes.sizeof(ModelInfoC)
        # Reasonable bounds for 64-bit systems
        assert 80 <= size <= 128, f"Unexpected struct size: {size} bytes"


class TestExecutionPlan:
    """Tests for ExecutionPlan class."""

    def test_execution_plan_construction(self):
        """ExecutionPlan can be constructed with all required parameters."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_bf16",
            attention_type="GroupedQueryAttention",
            ffn_type="SwiGLU(SiLU)",
            num_layers=28,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=64,
            num_experts=0,
            experts_per_token=0,
            quant_bits=16,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=False,
            uses_quantization=False,
            uses_gelu=False,
        )

        assert plan.matmul_kernel == "matmul_bf16"
        assert plan.attention_type == "GroupedQueryAttention"
        assert plan.ffn_type == "SwiGLU(SiLU)"
        assert plan.num_layers == 28
        assert plan.hidden_size == 1024
        assert plan.num_heads == 16
        assert plan.num_kv_heads == 8
        assert plan.head_dim == 64
        assert plan.num_experts == 0
        assert plan.experts_per_token == 0
        assert plan.quant_bits == 16
        assert plan.quant_group_size == 64
        assert plan.uses_gqa is True
        assert plan.uses_moe is False
        assert plan.uses_quantization is False
        assert plan.uses_gelu is False

    def test_execution_plan_quantized_model(self):
        """ExecutionPlan correctly identifies quantized models."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_grouped_affine_u4",
            attention_type="GroupedQueryAttention",
            ffn_type="SwiGLU(SiLU)",
            num_layers=28,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            num_experts=0,
            experts_per_token=0,
            quant_bits=4,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=False,
            uses_quantization=True,
            uses_gelu=False,
        )

        assert plan.uses_quantization is True
        assert plan.quant_bits == 4
        assert "matmul_grouped_affine_u4" in plan.matmul_kernel

    def test_execution_plan_moe_model(self):
        """ExecutionPlan correctly identifies MoE models."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_bf16",
            attention_type="GroupedQueryAttention",
            ffn_type="MoE(SiLU)",
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_experts=8,
            experts_per_token=2,
            quant_bits=16,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=True,
            uses_quantization=False,
            uses_gelu=False,
        )

        assert plan.uses_moe is True
        assert plan.num_experts == 8
        assert plan.experts_per_token == 2
        assert "MoE" in plan.ffn_type

    def test_repr_gqa_quantized(self):
        """ExecutionPlan repr shows GQA and quantization."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_grouped_affine_u4",
            attention_type="GroupedQueryAttention",
            ffn_type="SwiGLU(SiLU)",
            num_layers=28,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            num_experts=0,
            experts_per_token=0,
            quant_bits=4,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=False,
            uses_quantization=True,
            uses_gelu=False,
        )

        r = repr(plan)
        assert "matmul_grouped_affine_u4" in r
        assert "GQA" in r
        assert "Q4" in r

    def test_repr_moe(self):
        """ExecutionPlan repr shows MoE info."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_bf16",
            attention_type="GroupedQueryAttention",
            ffn_type="MoE(SiLU)",
            num_layers=32,
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_experts=8,
            experts_per_token=2,
            quant_bits=16,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=True,
            uses_quantization=False,
            uses_gelu=False,
        )

        r = repr(plan)
        assert "MoE(8)" in r

    def test_repr_fp16(self):
        """ExecutionPlan repr shows FP16 for non-quantized models."""
        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_bf16",
            attention_type="MultiHeadAttention",
            ffn_type="SwiGLU(SiLU)",
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            num_kv_heads=12,
            head_dim=64,
            num_experts=0,
            experts_per_token=0,
            quant_bits=16,
            quant_group_size=64,
            uses_gqa=False,
            uses_moe=False,
            uses_quantization=False,
            uses_gelu=False,
        )

        r = repr(plan)
        assert "FP16" in r

    def test_print_plan_does_not_raise(self):
        """ExecutionPlan.print_plan() does not raise."""
        import sys
        from io import StringIO

        from talu.converter import ExecutionPlan

        plan = ExecutionPlan(
            matmul_kernel="matmul_grouped_affine_u4",
            attention_type="GroupedQueryAttention",
            ffn_type="SwiGLU(SiLU)",
            num_layers=28,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            num_experts=0,
            experts_per_token=0,
            quant_bits=4,
            quant_group_size=64,
            uses_gqa=True,
            uses_moe=False,
            uses_quantization=True,
            uses_gelu=False,
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            plan.print_plan()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Verify output contains expected content
        assert "EXECUTION PLAN" in output
        assert "matmul_grouped_affine_u4" in output
        assert "GroupedQueryAttention" in output
        assert "SwiGLU(SiLU)" in output


class TestExecutionPlanFunction:
    """Tests for execution_plan() function.

    Note: Tests requiring actual model files are in tests/reference/models/.
    These tests verify API behavior without requiring model weights.
    """

    def test_execution_plan_raises_on_invalid_path(self):
        """execution_plan() raises ModelError for invalid paths."""
        from talu.converter import execution_plan
        from talu.exceptions import ModelError

        with pytest.raises(ModelError):
            execution_plan("/nonexistent/path/to/model")

    def test_execution_plan_raises_on_invalid_model_id(self):
        """execution_plan() raises ModelError for invalid model IDs."""
        from talu.converter import execution_plan
        from talu.exceptions import ModelError

        with pytest.raises(ModelError):
            execution_plan("definitely-not-a-real-org/not-a-real-model-12345")


class TestExecutionPlanInfoCStructLayout:
    """Tests that Python ExecutionPlanInfo matches Zig struct layout.

    These tests prevent ABI mismatches that cause segfaults.
    """

    def test_struct_has_expected_fields(self):
        """ExecutionPlanInfo has all expected fields in correct order."""
        import ctypes

        from talu._native import ExecutionPlanInfo

        # Expected fields in order (must match Zig struct exactly)
        expected_fields = [
            ("matmul_kernel", ctypes.c_char_p),
            ("attention_type", ctypes.c_char_p),
            ("ffn_type", ctypes.c_char_p),
            ("num_layers", ctypes.c_int32),
            ("hidden_size", ctypes.c_int32),
            ("num_heads", ctypes.c_int32),
            ("num_kv_heads", ctypes.c_int32),
            ("head_dim", ctypes.c_int32),
            ("num_experts", ctypes.c_int32),
            ("experts_per_token", ctypes.c_int32),
            ("quant_bits", ctypes.c_int32),
            ("quant_group_size", ctypes.c_int32),
            ("uses_gqa", ctypes.c_bool),
            ("uses_moe", ctypes.c_bool),
            ("uses_quantization", ctypes.c_bool),
            ("uses_gelu", ctypes.c_bool),
            ("is_supported", ctypes.c_bool),
            ("error_msg", ctypes.c_char_p),
        ]

        actual_fields = ExecutionPlanInfo._fields_

        assert len(actual_fields) == len(expected_fields), (
            f"Field count mismatch: expected {len(expected_fields)}, got {len(actual_fields)}"
        )

        for i, ((exp_name, exp_type), (act_name, act_type)) in enumerate(
            zip(expected_fields, actual_fields, strict=True)
        ):
            assert exp_name == act_name, (
                f"Field {i} name mismatch: expected '{exp_name}', got '{act_name}'"
            )
            assert exp_type == act_type, (
                f"Field '{exp_name}' type mismatch: expected {exp_type}, got {act_type}"
            )

    def test_error_msg_is_last_field(self):
        """error_msg must be the last field for error checking to work."""
        from talu._native import ExecutionPlanInfo

        fields = ExecutionPlanInfo._fields_
        last_field_name, _ = fields[-1]
        assert last_field_name == "error_msg", (
            f"Last field must be 'error_msg', got '{last_field_name}'"
        )
