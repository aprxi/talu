"""Reference tests for MiniLM embedding quality vs PyTorch.

Requires:
    - TALU_TEST_MINILM env var pointing to converted MiniLM-L6-v2-GAF8 model
    - sentence-transformers/all-MiniLM-L6-v2 available via transformers
"""

import math

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from talu.router import Router
from tests.conftest import TEST_MODEL_HF_MINILM, TEST_MODEL_URI_EMBEDDING


@pytest.fixture(scope="module")
def minilm_path():
    path = TEST_MODEL_URI_EMBEDDING
    if path is None:
        pytest.skip("MiniLM model not found. Set TALU_TEST_MINILM or place under models/.")
    return path


@pytest.fixture(scope="module")
def router(minilm_path):
    r = Router(models=[minilm_path])
    yield r
    r.close()


@pytest.fixture(scope="module")
def hf_model():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_HF_MINILM)
    model = AutoModel.from_pretrained(TEST_MODEL_HF_MINILM)
    model.eval()
    return tokenizer, model


def _hf_embed(tokenizer, model, text: str, pooling: str = "mean") -> np.ndarray:
    """Compute embedding using HuggingFace model."""
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [1, seq_len, 384]
        mask = inputs["attention_mask"]

    if pooling == "mean":
        expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
        emb = torch.sum(hidden * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
    elif pooling == "first":
        emb = hidden[:, 0, :]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    return emb[0].numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiniLMEmbedBasic:
    """Basic embedding functionality."""

    def test_embedding_dim_is_384(self, router):
        assert router.embedding_dim() == 384

    def test_embed_returns_384_floats(self, router):
        emb = router.embed("Hello world", pooling="mean")
        assert len(emb) == 384
        assert all(isinstance(x, float) for x in emb)

    def test_embed_normalized_unit_length(self, router):
        emb = router.embed("Hello world", normalize=True, pooling="mean")
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01

    def test_embed_unnormalized_has_nonunit_norm(self, router):
        emb = router.embed("Hello world", normalize=False, pooling="mean")
        norm = math.sqrt(sum(x * x for x in emb))
        assert norm > 1.0  # MiniLM unnormalized norms are typically 4-8

    def test_embed_deterministic(self, router):
        a = router.embed("Hello world", pooling="mean")
        b = router.embed("Hello world", pooling="mean")
        assert a == b

    def test_embed_different_texts_differ(self, router):
        a = router.embed("The weather is sunny today", pooling="mean")
        b = router.embed("Neural networks learn from data", pooling="mean")
        assert a != b

    def test_embed_pooling_strategies_differ(self, router):
        mean = router.embed("Hello world", pooling="mean", normalize=False)
        first = router.embed("Hello world", pooling="first", normalize=False)
        assert mean != first


class TestMiniLMEmbedAccuracy:
    """Numerical accuracy vs PyTorch reference.

    Tolerances account for GAF8 quantization. Full-precision models
    would allow tighter bounds (cosine > 0.9999).
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
        ],
        ids=["short", "long", "technical"],
    )
    def test_cosine_similarity_vs_pytorch(self, router, hf_model, text):
        """Talu embedding has cosine similarity > 0.999 vs PyTorch."""
        tokenizer, model = hf_model
        talu_emb = np.array(router.embed(text, normalize=False, pooling="mean"))
        hf_emb = _hf_embed(tokenizer, model, text, pooling="mean")

        cos_sim = np.dot(talu_emb, hf_emb) / (np.linalg.norm(talu_emb) * np.linalg.norm(hf_emb))
        assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} < 0.999 for '{text}'"

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
        ],
        ids=["short", "long"],
    )
    def test_norm_ratio_vs_pytorch(self, router, hf_model, text):
        """Talu embedding norm within 5% of PyTorch."""
        tokenizer, model = hf_model
        talu_emb = np.array(router.embed(text, normalize=False, pooling="mean"))
        hf_emb = _hf_embed(tokenizer, model, text, pooling="mean")

        ratio = np.linalg.norm(talu_emb) / np.linalg.norm(hf_emb)
        assert 0.95 < ratio < 1.05, f"Norm ratio {ratio:.4f} outside [0.95, 1.05]"

    def test_max_absolute_difference(self, router, hf_model):
        """Per-element max difference bounded for GAF8 quantized model."""
        tokenizer, model = hf_model
        text = "Hello world"
        talu_emb = np.array(router.embed(text, normalize=False, pooling="mean"))
        hf_emb = _hf_embed(tokenizer, model, text, pooling="mean")

        max_diff = np.max(np.abs(talu_emb - hf_emb))
        # GAF8 quantization introduces small errors; 0.05 is generous
        assert max_diff < 0.05, (
            f"Max abs diff {max_diff:.6f} >= 0.05 | "
            f"talu_norm={np.linalg.norm(talu_emb):.4f} "
            f"hf_norm={np.linalg.norm(hf_emb):.4f} "
            f"cosine={np.dot(talu_emb, hf_emb) / (np.linalg.norm(talu_emb) * np.linalg.norm(hf_emb)):.6f}"
        )

    def test_first_pooling_vs_pytorch(self, router, hf_model):
        """CLS-token (first) pooling matches PyTorch."""
        tokenizer, model = hf_model
        text = "Hello world"
        talu_emb = np.array(router.embed(text, normalize=False, pooling="first"))
        hf_emb = _hf_embed(tokenizer, model, text, pooling="first")

        cos_sim = np.dot(talu_emb, hf_emb) / (np.linalg.norm(talu_emb) * np.linalg.norm(hf_emb))
        assert cos_sim > 0.999, (
            f"CLS cosine similarity {cos_sim:.6f} < 0.999 | "
            f"talu_norm={np.linalg.norm(talu_emb):.4f} "
            f"hf_norm={np.linalg.norm(hf_emb):.4f} "
            f"max_diff={np.max(np.abs(talu_emb - hf_emb)):.6f}"
        )
