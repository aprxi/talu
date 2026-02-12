"""
Tests for synthetic model generation utilities.

Validates that model_factory produces valid artifacts:
- config.json matches expected schema and values
- tokenizer.json can be loaded by the Tokenizer class
- model.safetensors contains expected weight keys/shapes
"""

import json
import struct

import numpy as np

from helpers.model_factory import (
    create_minimal_config,
    create_minimal_model,
    create_minimal_tokenizer,
    create_minimal_weights,
    save_safetensors,
)


class TestCreateMinimalConfig:
    """Tests for config.json generation."""

    def test_config_has_required_fields(self):
        """Config contains all required LLM fields."""
        config = create_minimal_config()

        required_fields = [
            "architectures",
            "model_type",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
            "bos_token_id",
            "eos_token_id",
        ]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

    def test_config_values_match_parameters(self):
        """Config values match input parameters exactly."""
        config = create_minimal_config(
            vocab_size=500,
            hidden_size=128,
            num_layers=4,
            num_heads=8,
            intermediate_size=256,
            max_seq_len=1024,
            model_type="qwen",
        )

        assert config["vocab_size"] == 500
        assert config["hidden_size"] == 128
        assert config["num_hidden_layers"] == 4
        assert config["num_attention_heads"] == 8
        assert config["intermediate_size"] == 256
        assert config["max_position_embeddings"] == 1024
        assert config["model_type"] == "qwen"

    def test_config_architectures_format(self):
        """Architectures field is correctly formatted."""
        config = create_minimal_config(model_type="llama")
        assert config["architectures"] == ["LlamaForCausalLM"]

        config = create_minimal_config(model_type="qwen")
        assert config["architectures"] == ["QwenForCausalLM"]

    def test_config_has_special_token_ids(self):
        """Config includes BOS, EOS, and PAD token IDs."""
        config = create_minimal_config()

        assert config["bos_token_id"] == 1
        assert config["eos_token_id"] == 2
        assert config["pad_token_id"] == 0

    def test_config_serializable_to_json(self):
        """Config can be serialized to valid JSON."""
        config = create_minimal_config()
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        assert parsed == config


class TestCreateMinimalTokenizer:
    """Tests for tokenizer.json generation."""

    def test_tokenizer_has_required_structure(self):
        """Tokenizer has required top-level structure."""
        tokenizer = create_minimal_tokenizer()

        assert "version" in tokenizer
        assert "model" in tokenizer
        assert "added_tokens" in tokenizer

    def test_tokenizer_model_is_bpe(self):
        """Tokenizer model type is BPE."""
        tokenizer = create_minimal_tokenizer()

        assert tokenizer["model"]["type"] == "BPE"
        assert "vocab" in tokenizer["model"]
        assert "merges" in tokenizer["model"]

    def test_tokenizer_vocab_size_matches(self):
        """Vocabulary size matches requested size."""
        tokenizer = create_minimal_tokenizer(vocab_size=500)
        vocab = tokenizer["model"]["vocab"]

        # Vocab should have at most vocab_size entries
        assert len(vocab) <= 500

    def test_tokenizer_has_special_tokens(self):
        """Tokenizer includes special tokens."""
        tokenizer = create_minimal_tokenizer()
        vocab = tokenizer["model"]["vocab"]

        assert vocab["<pad>"] == 0
        assert vocab["<s>"] == 1
        assert vocab["</s>"] == 2
        assert vocab["<unk>"] == 3

    def test_tokenizer_added_tokens_match_vocab(self):
        """Added tokens match special token IDs in vocab."""
        tokenizer = create_minimal_tokenizer()
        vocab = tokenizer["model"]["vocab"]
        added = tokenizer["added_tokens"]

        # Build map of added token content -> id
        added_map = {t["content"]: t["id"] for t in added}

        assert added_map["<pad>"] == vocab["<pad>"]
        assert added_map["<s>"] == vocab["<s>"]
        assert added_map["</s>"] == vocab["</s>"]
        assert added_map["<unk>"] == vocab["<unk>"]

    def test_tokenizer_serializable_to_json(self):
        """Tokenizer can be serialized to valid JSON."""
        tokenizer = create_minimal_tokenizer()
        json_str = json.dumps(tokenizer)
        parsed = json.loads(json_str)
        assert parsed == tokenizer


class TestCreateMinimalWeights:
    """Tests for weight tensor generation."""

    def test_weights_has_embeddings(self):
        """Weights include embedding layer."""
        weights = create_minimal_weights(vocab_size=100, hidden_size=32)

        assert "model.embed_tokens.weight" in weights
        assert weights["model.embed_tokens.weight"].shape == (100, 32)

    def test_weights_has_layer_weights(self):
        """Weights include per-layer transformer weights."""
        weights = create_minimal_weights(
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
        )

        for i in range(2):
            prefix = f"model.layers.{i}"

            # Attention
            assert f"{prefix}.self_attn.q_proj.weight" in weights
            assert f"{prefix}.self_attn.k_proj.weight" in weights
            assert f"{prefix}.self_attn.v_proj.weight" in weights
            assert f"{prefix}.self_attn.o_proj.weight" in weights

            # FFN
            assert f"{prefix}.mlp.gate_proj.weight" in weights
            assert f"{prefix}.mlp.up_proj.weight" in weights
            assert f"{prefix}.mlp.down_proj.weight" in weights

            # Layer norms
            assert f"{prefix}.input_layernorm.weight" in weights
            assert f"{prefix}.post_attention_layernorm.weight" in weights

    def test_weights_has_final_norm(self):
        """Weights include final layer norm."""
        weights = create_minimal_weights()

        assert "model.norm.weight" in weights

    def test_weights_shapes_are_correct(self):
        """Weight tensor shapes match expected dimensions."""
        weights = create_minimal_weights(
            vocab_size=100,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
        )

        # Embeddings: [vocab_size, hidden_size]
        assert weights["model.embed_tokens.weight"].shape == (100, 32)

        # Attention: [hidden_size, hidden_size]
        assert weights["model.layers.0.self_attn.q_proj.weight"].shape == (32, 32)

        # FFN gate/up: [intermediate_size, hidden_size]
        assert weights["model.layers.0.mlp.gate_proj.weight"].shape == (64, 32)

        # FFN down: [hidden_size, intermediate_size]
        assert weights["model.layers.0.mlp.down_proj.weight"].shape == (32, 64)

        # Layer norms: [hidden_size]
        assert weights["model.layers.0.input_layernorm.weight"].shape == (32,)

    def test_weights_are_float32(self):
        """All weights are float32."""
        weights = create_minimal_weights()

        for name, arr in weights.items():
            assert arr.dtype == np.float32, f"{name} is not float32"

    def test_weights_are_reproducible(self):
        """Same seed produces identical weights."""
        weights1 = create_minimal_weights(seed=123)
        weights2 = create_minimal_weights(seed=123)

        for name in weights1:
            np.testing.assert_array_equal(weights1[name], weights2[name], err_msg=f"{name} differs")

    def test_weights_differ_with_different_seeds(self):
        """Different seeds produce different weights."""
        weights1 = create_minimal_weights(seed=1)
        weights2 = create_minimal_weights(seed=2)

        # At least embeddings should differ
        assert not np.allclose(
            weights1["model.embed_tokens.weight"],
            weights2["model.embed_tokens.weight"],
        )


class TestSaveSafetensors:
    """Tests for safetensors file writing."""

    def test_creates_valid_safetensors_file(self, tmp_path):
        """Creates a file with valid safetensors header."""
        weights = {"test": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        path = tmp_path / "test.safetensors"

        save_safetensors(path, weights)

        assert path.exists()

        # Read and validate header
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)

        assert "test" in header
        assert header["test"]["dtype"] == "F32"
        assert header["test"]["shape"] == [3]

    def test_safetensors_data_is_correct(self, tmp_path):
        """Tensor data in file matches input."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weights = {"matrix": arr}
        path = tmp_path / "test.safetensors"

        save_safetensors(path, weights)

        # Read back the data
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)

            start, end = header["matrix"]["data_offsets"]
            f.seek(8 + header_size + start)
            data = f.read(end - start)

        loaded = np.frombuffer(data, dtype=np.float32).reshape(2, 2)
        np.testing.assert_array_equal(loaded, arr)


class TestCreateMinimalModel:
    """Tests for complete model creation."""

    def test_creates_all_required_files(self):
        """Creates config.json, tokenizer.json, and model.safetensors."""
        with create_minimal_model() as model:
            assert (model.path / "config.json").exists()
            assert (model.path / "tokenizer.json").exists()
            assert (model.path / "model.safetensors").exists()

    def test_config_file_is_valid_json(self):
        """config.json is valid JSON with expected fields."""
        with create_minimal_model(vocab_size=500, hidden_size=64) as model:
            with open(model.path / "config.json") as f:
                config = json.load(f)

            assert config["vocab_size"] == 500
            assert config["hidden_size"] == 64

    def test_tokenizer_file_is_valid_json(self):
        """tokenizer.json is valid JSON with expected structure."""
        with create_minimal_model(vocab_size=500) as model:
            with open(model.path / "tokenizer.json") as f:
                tokenizer = json.load(f)

            assert tokenizer["model"]["type"] == "BPE"

    def test_model_attributes_match_parameters(self):
        """SyntheticModel attributes match creation parameters."""
        with create_minimal_model(
            vocab_size=200,
            hidden_size=32,
            num_layers=3,
            num_heads=4,
            intermediate_size=128,
            max_seq_len=256,
        ) as model:
            assert model.vocab_size == 200
            assert model.hidden_size == 32
            assert model.num_layers == 3
            assert model.num_heads == 4
            assert model.intermediate_size == 128
            assert model.max_seq_len == 256

    def test_model_weights_are_populated(self):
        """Model weights dictionary is populated."""
        with create_minimal_model() as model:
            assert len(model.weights) > 0
            assert "model.embed_tokens.weight" in model.weights

    def test_context_manager_cleanup(self, tmp_path):
        """Context manager cleans up temp directory."""
        # Create model and capture path
        with create_minimal_model() as model:
            model_path = model.path
            assert model_path.exists()

        # After context exit, temp dir should be cleaned up
        assert not model_path.exists()

    def test_explicit_output_dir_not_cleaned(self, tmp_path):
        """Explicit output_dir is not cleaned up."""
        output = tmp_path / "my_model"

        with create_minimal_model(output_dir=output) as model:
            assert model.path == output

        # Should still exist after context exit
        assert output.exists()
        assert (output / "config.json").exists()

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical weights."""
        with create_minimal_model(seed=42) as model1:
            weights1 = {k: v.copy() for k, v in model1.weights.items()}

        with create_minimal_model(seed=42) as model2:
            weights2 = model2.weights

        for name in weights1:
            np.testing.assert_array_equal(weights1[name], weights2[name])


class TestTokenizerIntegration:
    """Tests that synthetic tokenizer works with talu.Tokenizer."""

    def test_tokenizer_loads_synthetic_model(self):
        """Tokenizer can load synthetic tokenizer.json."""
        from talu import Tokenizer

        with create_minimal_model() as model:
            tokenizer = Tokenizer(str(model.path))

            # Basic sanity check
            assert tokenizer.vocab_size > 0

    def test_tokenizer_encodes_text(self):
        """Tokenizer can encode text with synthetic vocab."""
        from talu import Tokenizer

        with create_minimal_model(vocab_size=1000) as model:
            tokenizer = Tokenizer(str(model.path))

            # Encode simple ASCII text (should work with our byte-level vocab)
            tokens = tokenizer.encode("hello")
            assert len(tokens) > 0

    def test_tokenizer_decodes_tokens(self):
        """Tokenizer can decode tokens back to text."""
        from talu import Tokenizer

        with create_minimal_model(vocab_size=1000) as model:
            tokenizer = Tokenizer(str(model.path))

            # Round-trip: encode then decode
            original = "test"
            tokens = tokenizer.encode(original)
            decoded = tokenizer.decode(tokens)

            # With byte-level BPE, should get back the original
            assert decoded == original


class TestModelFactoryValidation:
    """Tests for model factory input validation via synthetic_model_factory."""

    def test_factory_validates_vocab_size(self):
        """Factory rejects vocab_size < 100."""
        import pytest

        from tests.reference.fixtures import synthetic_model_factory

        with pytest.raises(ValueError, match="vocab_size must be >= 100"):
            synthetic_model_factory(vocab_size=50)

    def test_factory_validates_num_heads(self):
        """Factory rejects num_heads < 1."""
        import pytest

        from tests.reference.fixtures import synthetic_model_factory

        with pytest.raises(ValueError, match="num_heads must be >= 1"):
            synthetic_model_factory(num_heads=0)

    def test_factory_validates_hidden_size_divisibility(self):
        """Factory rejects hidden_size not divisible by num_heads."""
        import pytest

        from tests.reference.fixtures import synthetic_model_factory

        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            synthetic_model_factory(hidden_size=65, num_heads=4)

    def test_factory_validates_num_layers(self):
        """Factory rejects num_layers < 1."""
        import pytest

        from tests.reference.fixtures import synthetic_model_factory

        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            synthetic_model_factory(num_layers=0)

    def test_factory_validates_intermediate_size(self):
        """Factory rejects intermediate_size < hidden_size."""
        import pytest

        from tests.reference.fixtures import synthetic_model_factory

        with pytest.raises(ValueError, match="intermediate_size.*should be >= hidden_size"):
            synthetic_model_factory(hidden_size=128, intermediate_size=64)


class TestWeightInvariants:
    """Tests for weight tensor invariants."""

    def test_embedding_weight_shape_invariant(self):
        """Embedding weight shape matches [vocab_size, hidden_size]."""
        weights = create_minimal_weights(vocab_size=500, hidden_size=128)
        emb = weights["model.embed_tokens.weight"]

        assert emb.shape[0] == 500, "First dim should be vocab_size"
        assert emb.shape[1] == 128, "Second dim should be hidden_size"

    def test_attention_weight_shape_invariant(self):
        """Attention projection weights are square [hidden_size, hidden_size]."""
        weights = create_minimal_weights(hidden_size=64, num_layers=1)

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = weights[f"model.layers.0.self_attn.{proj}.weight"]
            assert w.shape[0] == w.shape[1] == 64, f"{proj} should be square"

    def test_ffn_gate_up_shape_invariant(self):
        """Gate and up projections are [intermediate_size, hidden_size]."""
        weights = create_minimal_weights(hidden_size=64, intermediate_size=256, num_layers=1)

        for proj in ["gate_proj", "up_proj"]:
            w = weights[f"model.layers.0.mlp.{proj}.weight"]
            assert w.shape == (256, 64), f"{proj} shape mismatch"

    def test_ffn_down_shape_invariant(self):
        """Down projection is [hidden_size, intermediate_size]."""
        weights = create_minimal_weights(hidden_size=64, intermediate_size=256, num_layers=1)

        w = weights["model.layers.0.mlp.down_proj.weight"]
        assert w.shape == (64, 256), "down_proj shape mismatch"

    def test_layer_norm_shape_invariant(self):
        """Layer norm weights are 1D [hidden_size]."""
        weights = create_minimal_weights(hidden_size=64, num_layers=1)

        for ln in ["input_layernorm", "post_attention_layernorm"]:
            w = weights[f"model.layers.0.{ln}.weight"]
            assert w.shape == (64,), f"{ln} should be 1D"
            assert w.ndim == 1, f"{ln} should have 1 dimension"

    def test_dtype_consistency_invariant(self):
        """All weights have consistent dtype (float32)."""
        weights = create_minimal_weights()

        for name, arr in weights.items():
            assert arr.dtype == np.float32, f"{name} has dtype {arr.dtype}, expected float32"

    def test_weight_name_consistency_invariant(self):
        """Weight names follow expected naming convention."""
        weights = create_minimal_weights(num_layers=2)

        # All layer weights should have consistent layer indices
        for i in range(2):
            prefix = f"model.layers.{i}."
            layer_weights = [k for k in weights if k.startswith(prefix)]
            assert len(layer_weights) == 9, f"Layer {i} should have 9 weight tensors"
