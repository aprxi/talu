# CI Synthetic Model Fallback

## Why
- CI runners typically do not have cached HuggingFace models, so `@pytest.mark.requires_model` tests skip.
- Skipped tests mean we lose coverage for core behavior on clean environments.
- A tiny, local synthetic model keeps tests runnable without network access or large downloads.

## How
- Create a session-scoped pytest fixture that builds a minimal synthetic model (config + tokenizer + small random weights) using `tests/helpers/model_factory.py`.
- Update model-dependent fixtures to fallback to the synthetic model when a real model is not available.
- Keep the real-model path as the first choice; use synthetic only as fallback.

## What
- Add a fixture (e.g., `synthetic_model_path`) in `tests/conftest.py`.
- Update `test_model_path` (and any similar fixtures) to return the synthetic model when no real model is found.
- Ensure tokenizer and converter tests can use the synthetic files when models are missing.
- Add a short note in test docs explaining the fallback behavior and CI expectations.
