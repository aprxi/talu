import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Path: bindings/python/tests/reference/graph/run_graph_case.py -> parents[5] = repo root
ROOT = Path(__file__).resolve().parents[5]
MODELS_ROOT = ROOT / "models"
# tests/reference contains the 'graph' package (graph/utils.py)
TESTS_REFERENCE_ROOT = ROOT / "bindings" / "python" / "tests" / "reference"
# Ensure ROOT is first so 'import models' finds models/ not tests/models/.
# Remove existing entries to control order.
for path in [str(ROOT), str(MODELS_ROOT), str(TESTS_REFERENCE_ROOT)]:
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(TESTS_REFERENCE_ROOT))
sys.path.insert(0, str(MODELS_ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from helpers.model_factory import create_minimal_model, save_safetensors  # noqa: E402

from graph.utils import (  # noqa: E402
    block_kwargs_for_arch,
    build_block_weights,
    get_block_class,
    write_graph_for_arch,
)


def _run_generation(model_path: str, prompt: str, max_tokens: int) -> list[int]:
    """
    Run generation using the Router C API with streaming to capture token IDs.

    Uses FusedCpuBackend for production inference with:
    - Fused graph operations (norm, attn, ffn combined)
    - Optimized SIMD kernels

    Note: This uses raw prompt directly (no chat template).
    """
    import ctypes

    from talu._bindings import get_lib
    from talu._native import BackendCreateOptions, ChatCreateOptions
    from talu.router import normalize_to_handle
    from talu.router._bindings import (
        CContentPart,
        RouterGenerateConfig,
        TaluInferenceBackendHandle,
        get_spec_lib,
    )

    # Signatures are set up automatically by _native.py at import time
    lib = get_lib()
    spec_lib = get_spec_lib()

    # Create chat handle
    options = ChatCreateOptions(offline=False)
    chat_ptr = lib.talu_chat_create(ctypes.byref(options))
    if not chat_ptr:
        raise RuntimeError("Failed to create chat")

    # Create backend from model path (normalize_to_handle accepts string path directly)
    canonical_handle = normalize_to_handle(model_path)
    backend_handle = TaluInferenceBackendHandle()
    options = BackendCreateOptions()
    code = spec_lib.talu_backend_create_from_canonical(
        canonical_handle, options, ctypes.byref(backend_handle)
    )
    if code != 0:
        from talu._bindings import get_last_error

        error = get_last_error() or f"Unknown error code: {code}"
        raise RuntimeError(f"Failed to create backend: {error}")

    # Callback type: (token_text, token_id, user_data) -> continue?
    RouterTokenCallback = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )

    # Collect token IDs from streaming callback
    collected_tokens: list[int] = []

    @RouterTokenCallback
    def token_callback(token_text, token_id, user_data):
        collected_tokens.append(token_id)
        return True  # Continue generation

    try:
        # Note: We don't add the user message here - talu_router_stream adds it internally
        # See capi/router.zig

        # Build config
        config = RouterGenerateConfig(
            max_tokens=max_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            stop_sequences=None,
            logit_bias=None,
            seed=0,
        )

        # Build content part for the prompt (text type = 0)
        prompt_bytes = prompt.encode("utf-8")
        prompt_buffer = ctypes.create_string_buffer(prompt_bytes)
        content_part = CContentPart(
            content_type=0,  # text
            _padding=(ctypes.c_uint8 * 7)(),
            data_ptr=ctypes.cast(prompt_buffer, ctypes.POINTER(ctypes.c_char)),
            data_len=len(prompt_bytes),
            mime_ptr=None,
        )
        parts_array = (CContentPart * 1)(content_part)

        # Call streaming API with backend to get token IDs via callback
        result = lib.talu_router_stream_with_backend(
            chat_ptr,
            parts_array,
            1,  # num_parts
            backend_handle,
            ctypes.byref(config),
            token_callback,
            None,  # user_data
        )

        if result.error_code != 0:
            from talu._bindings import get_last_error

            error = get_last_error() or f"Unknown error code: {result.error_code}"
            raise RuntimeError(f"Generation failed: {error}")

        # Free result (text is allocated by Zig)
        lib.talu_router_result_free(ctypes.byref(result))

        return collected_tokens

    finally:
        lib.talu_chat_free(chat_ptr)
        spec_lib.talu_backend_free(backend_handle)
        spec_lib.talu_config_free(canonical_handle)


ARCH_CONFIGS = {
    "llama2": {
        "model_type": "llama2_test",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
    },
    "llama3": {
        "model_type": "llama3",
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
    },
    "qwen3": {
        "model_type": "qwen3",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    },
    "gemma3": {
        "model_type": "gemma3",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh",
        "use_qk_norm": True,
    },
    "phi4": {
        "model_type": "phi4",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "partial_rotary_factor": 1.0,
    },
    "granite3": {
        "model_type": "granite",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["graph"], required=True)
    parser.add_argument("--primitive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", default="abc")
    parser.add_argument("--arch", choices=sorted(ARCH_CONFIGS.keys()), default="llama2")
    args = parser.parse_args()

    arch_cfg = ARCH_CONFIGS[args.arch]
    model_type = arch_cfg["model_type"]

    # Validate seed is applied consistently to all random sources
    # This ensures deterministic behavior across runs
    assert args.seed is not None, "Seed must be specified for reproducible tests"

    with tempfile.TemporaryDirectory(prefix="talu_graphs_") as graphs_dir:
        with create_minimal_model(
            vocab_size=32,
            hidden_size=8,
            num_layers=1,
            num_heads=1,
            intermediate_size=16,
            max_seq_len=64,  # Must be > prompt length after chat template expansion
            model_type=model_type,
            seed=args.seed,
        ) as model:
            # Create the RNG instance used for test weight generation.
            # This is the same RNG that will be used for embeddings below.
            rng = np.random.default_rng(args.seed)

            # Verify seeding is effective by sampling and comparing against
            # a fresh RNG with the same seed. This catches cases where the
            # seed is not applied correctly to np.random.default_rng.
            _rng_sample = rng.random()
            _rng_verify = np.random.default_rng(args.seed).random()

            assert _rng_sample == _rng_verify, (
                f"NumPy default_rng seeding failed: seed={args.seed} produced "
                f"{_rng_sample} vs fresh RNG {_rng_verify}"
            )

            # Re-create RNG after verification sample (consumed one value)
            rng = np.random.default_rng(args.seed)

            # Also verify torch seeding for any torch-based randomness
            torch.manual_seed(args.seed)
            _torch_sample = torch.rand(1).item()
            torch.manual_seed(args.seed)
            _torch_verify = torch.rand(1).item()

            assert _torch_sample == _torch_verify, (
                f"PyTorch RNG seeding failed: seed={args.seed} produced "
                f"{_torch_sample} then {_torch_verify}"
            )

            hidden = model.hidden_size
            vocab = model.vocab_size

            config = {
                "architectures": [f"{model_type.title()}ForCausalLM"],
                "model_type": model_type,
                "hidden_size": model.hidden_size,
                "num_attention_heads": model.num_heads,
                "num_key_value_heads": model.num_heads,
                "intermediate_size": model.intermediate_size,
                "max_position_embeddings": model.max_seq_len,
                "num_hidden_layers": model.num_layers,
                "vocab_size": model.vocab_size,
                "tie_word_embeddings": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            }
            config.update(arch_cfg)

            with open(model.path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            block_class = get_block_class(args.arch)
            block_cfg = block_kwargs_for_arch(args.arch, config)

            # Use deterministic random weights (not zeros!) so the model
            # actually computes something meaningful.
            # Note: build_block_weights creates its own RNG from seed for isolation.
            weights = build_block_weights(block_class, block_cfg, "model.layers.0.", seed=args.seed)

            # Random embeddings using our verified RNG (scaled appropriately)
            emb = rng.normal(0, 0.02, size=(vocab, hidden)).astype(np.float32)
            weights["model.embed_tokens.weight"] = emb
            weights["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
            save_safetensors(model.path / "model.safetensors", weights)
            model.weights = weights

            write_graph_for_arch(Path(graphs_dir), args.arch, model_type, block_cfg, args.primitive)

            os.environ["TALU_GRAPHS_PATH"] = graphs_dir
            os.environ["BACKEND"] = "cpu"

            tokens = _run_generation(
                str(model.path),
                args.prompt,
                args.max_tokens,
            )

            result = {"tokens": tokens}
            print(json.dumps(result))


if __name__ == "__main__":
    main()
