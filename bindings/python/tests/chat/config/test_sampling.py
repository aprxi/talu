"""
Tests for SamplingParams and SamplingStrategy.

Tests the low-level sampling configuration types:
- SamplingStrategy enum (GREEDY, TOP_K, TOP_P)
- SamplingParams ctypes struct
- Integration with GenerationConfig._to_sampling_params()
"""

from talu.router import SamplingParams, SamplingStrategy


class TestSamplingStrategyEnum:
    """Tests for SamplingStrategy enum constants."""

    def test_greedy_value(self):
        """GREEDY has value 0."""
        assert SamplingStrategy.GREEDY == 0

    def test_top_k_value(self):
        """TOP_K has value 1."""
        assert SamplingStrategy.TOP_K == 1

    def test_top_p_value(self):
        """TOP_P has value 2."""
        assert SamplingStrategy.TOP_P == 2


class TestSamplingParamsCreation:
    """Tests for SamplingParams struct creation."""

    def test_default_values(self):
        """SamplingParams has expected defaults."""
        params = SamplingParams()
        assert params.strategy == SamplingStrategy.GREEDY
        assert params.temperature == 1.0
        assert params.top_k == 50
        assert abs(params.top_p - 0.9) < 1e-5
        assert params.min_p == 0.0
        assert params.repetition_penalty == 1.0
        assert params.seed == 0

    def test_custom_values(self):
        """SamplingParams accepts custom values."""
        params = SamplingParams(
            strategy=SamplingStrategy.TOP_K,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.2,
            seed=42,
        )
        assert params.strategy == SamplingStrategy.TOP_K
        assert abs(params.temperature - 0.7) < 1e-5
        assert params.top_k == 40
        assert abs(params.top_p - 0.95) < 1e-5
        assert abs(params.min_p - 0.05) < 1e-5
        assert abs(params.repetition_penalty - 1.2) < 1e-5
        assert params.seed == 42


class TestSamplingParamsStructLayout:
    """Tests for SamplingParams ctypes struct layout."""

    def test_struct_can_be_passed_to_c(self):
        """SamplingParams struct can be passed to C (no exceptions)."""
        params = SamplingParams(
            strategy=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.2,
        )
        # Verify struct can be passed to C (no exceptions)
        assert params.strategy == 1
        assert abs(params.min_p - 0.05) < 1e-5
        assert abs(params.repetition_penalty - 1.2) < 1e-5

    def test_struct_fields_accessible(self):
        """All struct fields are accessible."""
        params = SamplingParams()
        # Should not raise
        _ = params.strategy
        _ = params.temperature
        _ = params.top_k
        _ = params.top_p
        _ = params.min_p
        _ = params.repetition_penalty
        _ = params.seed


class TestSamplingParamsIntegration:
    """Tests for SamplingParams integration with GenerationConfig."""

    def test_generation_config__to_sampling_params(self):
        """GenerationConfig._to_sampling_params() returns SamplingParams."""
        from talu.router import GenerationConfig

        config = GenerationConfig(
            temperature=0.8,
            top_k=60,
            top_p=0.85,
            min_p=0.1,
            repetition_penalty=1.15,
        )
        params = config._to_sampling_params()

        assert isinstance(params, SamplingParams)
        assert abs(params.temperature - 0.8) < 1e-5
        assert params.top_k == 60
        assert abs(params.top_p - 0.85) < 1e-5
        assert abs(params.min_p - 0.1) < 1e-5
        assert abs(params.repetition_penalty - 1.15) < 1e-5

    def test_greedy_decoding_strategy(self):
        """Temperature 0.0 sets strategy to GREEDY."""
        from talu.router import GenerationConfig

        config = GenerationConfig(temperature=0.0)
        params = config._to_sampling_params()
        assert params.strategy == SamplingStrategy.GREEDY

    def test_sampling_strategy(self):
        """Non-zero temperature sets strategy to TOP_K."""
        from talu.router import GenerationConfig

        config = GenerationConfig(temperature=0.7)
        params = config._to_sampling_params()
        assert params.strategy == SamplingStrategy.TOP_K
