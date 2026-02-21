# Model Onboarding Checklist

Use this checklist when adding or updating model architecture metadata.

## Required Changes

1. Add/update architecture metadata in `core/src/models/<family>/*.zig`.
2. Register all model_type aliases in `core/src/models/registry.zig`.
3. Ensure metadata completeness in architecture definitions:
   - `d_ff_source_weight_ids` for MLP/MoE models.
   - `shortconv_dims_source_weight_id` for ShortConv models.
   - `force_f32` and transform flags on `WeightSpec` where required.
4. Keep runtime orchestration model-agnostic:
   - Avoid model-family logic in `core/src/inference/*`.
   - Use models-owned contracts (`core/src/models/op_types.zig`, `core/src/models/load/runtime_blocks.zig`).

## Required Tests

1. Update/add tests under `core/tests/models/*` for new metadata behavior.
2. Ensure onboarding contract tests pass for all aliases:
   - `core/tests/models/onboarding_contract_test.zig`
3. Ensure metadata integrity tests pass:
   - `core/tests/models/metadata_contract_test.zig`

## Required Gates

1. `zig build lint`
2. `zig build test-lint`
3. `zig build test-model-policy`
4. `zig build test -Drelease`
5. `zig build test-integration -Drelease`
6. `zig build release -Drelease`
7. `zig build models-report -- registry`
8. `zig build models-report -- metadata`

## Model Change Policy Tool

Use `zig build model-policy` to enforce model-change hygiene.

Examples:

```bash
zig build model-policy -- core/src/models/llama/llama3.zig core/tests/models/onboarding_contract_test.zig
```

When a model metadata change intentionally requires inference changes, set:

```bash
TALU_MODEL_INFERENCE_CHANGE_REASON="explain the runtime reason" \
zig build model-policy -- core/src/models/llama/llama3.zig core/src/inference/backend/cpu/engine.zig core/tests/models/onboarding_contract_test.zig
```

For CI, you can pass changed files via `TALU_CHANGED_FILES` (newline-separated) and run `zig build model-policy` without args.
