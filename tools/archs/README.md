# Adding New Model Architectures

This directory contains pure PyTorch reference implementations of model
architectures. These references serve two purposes:

1. **Correctness verification** -- run the model with PyTorch to confirm
   it produces sensible output before any Zig work begins.
2. **Graph generation** -- `trace.py` uses `torch.fx` to trace the
   `Block.forward()` and emit a JSON compute graph that the Zig runtime
   loads at build time.

## Directory layout

```
tools/archs/
  lib/               # Shared utilities (@fusable, loader, config, capture)
  _graphs/           # Generated JSON graphs (checked in)
  _reference/        # Reference NPZ files for debugging (generated, not checked in)
  tests/             # PyTorch reference test scripts
  <family>/          # One directory per model family
    __init__.py
    <arch>.py        # Pure PyTorch Block + full model
  trace.py           # torch.fx tracer + architecture registry
  capture.py         # Reference tensor capture for debugging
  compare.py         # Compare reference vs talu NPZ files
```

---

## Step 1: PyTorch reference (must produce correct output)

Everything in Step 1 leads to a working, testable PyTorch model that
generates coherent text. Do not proceed to Step 2 until this works.

### Step 1a: Study the HuggingFace model

Before writing any code, inspect the model you want to add:

```bash
# Download just config.json
python -c "
from huggingface_hub import hf_hub_download
import json
p = hf_hub_download('LiquidAI/LFM2-350M', 'config.json')
print(json.dumps(json.load(open(p)), indent=2))
"
```

Note the `model_type` field -- this is how talu detects which architecture
to use at runtime. Also note key dimensions: `hidden_size`,
`num_attention_heads`, `num_key_value_heads`, `intermediate_size`,
`num_hidden_layers`, and any architecture-specific fields.

Check the HuggingFace model page and source code (often in `transformers`)
to understand the block structure: norm placement, attention variant,
activation function, weight naming, etc.

### Step 1b: Write a pure PyTorch implementation

Create `<family>/<arch>.py` with:

1. **Building blocks** -- `RMSNorm`, `Attention`, `MLP`, etc.
2. **`Block` class** -- a single transformer block (this is what gets traced).
3. **Full model class** -- embedding + layers + norm + lm_head.
4. **`from_pretrained`** -- loads real HuggingFace weights.

Mark each fused block with `@fusable`:

```python
from lib.nn import fusable
from lib.utils import from_pretrained

@fusable(kernel="norm")
class RMSNorm(nn.Module): ...

@fusable(kernel="attention")           # or config=["qk_norm"] if applicable
class Attention(nn.Module): ...

@fusable(kernel="mlp")
class MLP(nn.Module): ...

class Block(nn.Module):
    """Single transformer block -- this is traced by torch.fx."""
    def __init__(self, hidden_size, num_heads, ...):
        ...
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class MyModel(nn.Module):
    def __init__(self, config):
        ...
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(model_id):
        return from_pretrained(MyModel, model_id)
```

#### Key conventions

- `Block.__init__` receives **scalar config values** (hidden_size, num_heads,
  etc.), not a dict. This is required for `torch.fx` tracing.
- For **heterogeneous** models (different layer types), use `HybridBlock(config, layer_idx)`
  instead -- see `lfm2/lfm2.py` or `granite/granite_hybrid.py`.
- The `@fusable(kernel=...)` decorator tells the tracer to treat the module
  as an opaque fused op. Available kernels: `norm`, `attention`, `mlp`,
  `shortconv`, `mamba_mixer`.
- Use `@fusable(kernel="attention", config=["qk_norm"])` to flag QK
  normalization so the Zig runtime allocates the right kernel.

#### Weight naming

The PyTorch parameter names in your `Block` must match the HuggingFace
checkpoint (minus the layer prefix). The tracer extracts weight specs
from the module's `state_dict()`.

Common pattern: HF stores `model.layers.0.self_attn.q_proj.weight` --
your Block should have `self.self_attn.q_proj` as an `nn.Linear`.

### Step 1c: Write a test script

Create `tests/test_<arch>.py`:

```python
#!/usr/bin/env python3
"""Test <arch> pure PyTorch implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from <family>.<arch> import MyModel

MODEL_ID = "org/model-name"

def generate(model, tokenizer, prompt, max_tokens=10):
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor([tokenizer.encode(text)])

    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(ids)
            next_id = logits[0, -1].argmax().item()
            if next_id == tokenizer.eos_token_id:
                break
            print(tokenizer.decode([next_id]), end="", flush=True)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    print()

def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = MyModel.from_pretrained(MODEL_ID)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    generate(model, tokenizer, prompt)

if __name__ == "__main__":
    main()
```

### Step 1d: Verify correctness

```bash
cd tools/archs
uv run python tests/test_<arch>.py
```

This **must** produce coherent text. If it doesn't, the PyTorch reference
has a bug -- fix it before proceeding. The Zig runtime cannot be correct
if the reference isn't.

### Step 1e: Register in trace.py

Add an entry to the `ARCHITECTURES` dict in `trace.py`:

```python
"myarch": {
    "name": "myarch",                    # Architecture ID
    "model_types": ["mytype"],           # config.json model_type values
    "module": "family.arch",             # Python import path
    "block_class": "Block",              # Class to trace
    "model_ids": ["org/model-name"],     # Default HF model for tracing
    "eps": 1e-5,                         # Norm epsilon
},
```

For **heterogeneous** models, also set:

```python
    "heterogeneous": True,
    "layer_types_key": "full_attn_idxs",  # or "layer_types"
```

Optional fields:

| Field | Default | Purpose |
|---|---|---|
| `embedding_scale` | 1.0 | Post-embedding multiplier (Gemma uses ~45.25) |
| `weight_prefixes` | standard | Custom block weight prefixes (BERT) |
| `pre_block` | standard embedding | Custom pre-block ops (BERT uses 3 embeddings) |
| `post_block` | standard norm+lm_head | Custom post-block ops |
| `global_weights` | standard | Custom global weight specs |

### Step 1f: Generate the compute graph

```bash
cd tools/archs
uv run python -m trace myarch
# Output: _graphs/myarch.json
```

Inspect the JSON to verify it looks correct -- check that ops, weights,
and weight_prefixes match what you expect.

### Step 1g: Generate reference NPZ (for debugging)

Generate a reference tensor capture that can be compared against talu:

```bash
cd tools/archs
uv run python -m capture myarch "Hello" --output _reference/myarch.npz
```

This runs the PyTorch reference with hooks at each trace point and saves
all intermediate tensors to NPZ. The trace points match talu's `trace.emit()`
points:

| Trace Point | Description |
|-------------|-------------|
| `embed` | After embedding lookup |
| `layer{N}.attn_norm` | After attention input norm |
| `layer{N}.attn_out` | After attention output projection |
| `layer{N}.ffn_norm` | After FFN input norm |
| `layer{N}.ffn_down` | After FFN down projection |
| `layer{N}.block_out` | After residual add |
| `final_norm` | After final layer norm |
| `lm_head` | Final logits |

The reference NPZ is used in Step 3 when debugging talu integration issues.

---

## Step 2: Zig integration

Rebuild to embed the new graph and verify detection:

```bash
zig build release -Drelease
```

The build system automatically embeds all `_graphs/*.json` files into
the Zig binary. After rebuilding, the new architecture is available:

```bash
./zig-out/bin/talu describe org/model-name
```

If the model uses only ops that the Zig runtime already supports (norm,
attention, mlp, shortconv, mamba), it should work without any Zig code
changes. If it requires a new op type, that's a separate task.

---

## Step 3: Debug (when talu output doesn't match)

If talu produces garbage or incorrect output, use the reference NPZ from
Step 1g to find the first divergence. **Run talu only once**, then debug
offline by comparing NPZ files.

### Step 3 checklist (required for new architectures)

For each new model/architecture being integrated:

1. Produce a PyTorch reference NPZ.
2. Produce a talu NPZ with `talu-dump`.
3. Compare NPZ files and identify the first divergence.
4. Fix the owning loader/kernel path.
5. Re-run compare until divergence moves forward (or disappears).

This is required even if text output is obviously wrong: tensor parity is
the shortest path to root cause.

### Step 3a: Export talu tensors to NPZ

Run talu with full tensor capture (one time only):

```bash
# Build the dump binary (requires release mode)
zig build dump -Drelease

# Run with tensor capture
./zig-out/bin/talu-dump -m org/model-name -p "Hello" -o /tmp/talu.npz
```

The `talu-dump` binary captures all intermediate tensors at each trace point
during inference and writes them to NPZ format. This is a Zig-native tool
that doesn't require Python bindings.

Options:
- `-m, --model <path>` -- Model path (required)
- `-p, --prompt <text>` -- Input prompt (default: "Hello")
- `-o, --output <path>` -- Output NPZ path (default: /tmp/talu.npz)
- `-n, --tokens <n>` -- Max tokens to generate (default: 1)
- `-l, --layer <N>` -- Capture only layer `N`
- `-l, --layer <A:B>` -- Capture layer range `A` through `B` (inclusive)
- `--point <name>` -- Capture only points containing `<name>` (repeatable)
- `-s, --stop-after-layer <N>` -- Stop execution after layer `N` (major runtime win)

Useful dumped layer points now include:
- `layer_attn_norm`
- `attn.out`
- `layer_ffn_norm`
- `ffn.down`
- `block.out`

Filtering options (for faster debugging):
- `-l, --layer <N>` -- Capture only layer N
- `-l, --layer <A:B>` -- Capture layers A through B (inclusive)
- `--point <name>` -- Capture only points containing `<name>` (repeatable)
- `-s, --stop-after-layer <N>` -- Stop execution after layer N (big runtime win)

Examples:
```bash
# Full dump
./zig-out/bin/talu-dump -m model -p "Hello" -o /tmp/full.npz

# Only layer 5 (fast inspection)
./zig-out/bin/talu-dump -m model -l 5 -o /tmp/layer5.npz

# Layers 0-3 with early stop (very fast)
./zig-out/bin/talu-dump -m model -l 0:3 -s 3 -o /tmp/first4.npz

# Only FFN norms across all layers
./zig-out/bin/talu-dump -m model --point ffn_norm -o /tmp/ffn_norms.npz
```

### Step 3b: Compare against reference

Compare the talu NPZ against the PyTorch reference NPZ:

```bash
cd tools/archs
uv run python -m compare _reference/myarch.npz /tmp/talu.npz
```

Output shows the first divergence:

```
✓ embed:              max_diff=1.2e-6
✓ layer0.attn_norm:   max_diff=3.4e-6
✓ layer0.attn_out:    max_diff=8.9e-6
✗ layer0.ffn_down:    max_diff=0.034   ← FIRST DIVERGENCE
  talu:  [0.023, -0.041, 0.018, ...]
  ref:   [0.024, -0.042, 0.019, ...]
```

Now you know exactly where to look: layer 0 FFN down projection.

For fast triage on large models, start with one layer:

```bash
./zig-out/bin/talu-dump -m org/model-name -p "Hello" -l 0 -s 0 -o /tmp/talu_l0.npz
```

Then compare and determine the first bad point in that layer (`attn.out` vs
`ffn.down`), before running wider layer ranges.

### Optional: one-command diagnosis

You can run capture + dump + compare in one command:

```bash
cd tools/archs
uv run python -m diagnose myarch --prompt "Hello"
```

Useful options:
- `--model-id` set reference model ID/path for PyTorch capture.
- `--talu-model` set model path/ID for `talu-dump`.
- `--threshold` set compare tolerance.
- `--skip-capture` / `--skip-dump` reuse existing NPZ artifacts.

### Step 3c: Targeted debugging

Once you know the divergent layer/point, you can:

1. **Check weights**: Compare talu's loaded weights vs PyTorch's for that layer
2. **Check kernel**: Verify the FFN kernel produces correct output
3. **Check shapes**: Ensure tensor shapes match at that point

### Why this workflow matters

Without this workflow, debugging requires repeated model runs:

```
❌ Old way (slow):
Run → garbage → hypothesis → run again → still garbage → new hypothesis → run again...
```

With NPZ comparison:

```
✓ New way (fast):
Run talu ONCE → export NPZ → compare offline → find divergence → fix → run once to verify
```

For a 30B model, each run takes significant time. The NPZ comparison is
instant numpy operations -- you can iterate on hypotheses without waiting
for model runs.

### Lessons that generalize across model adds

1. **Prompt/token parity is mandatory**
   Use the same prompt path for reference and talu (including chat-template behavior).
   If inputs differ, tensor diffs are not actionable.
2. **Check numerical health before quality**
   Confirm no NaN/Inf first. Semantic quality debugging comes after numeric stability.
3. **Debug by first divergence, not final text**
   Start at the earliest mismatched tensor; downstream noise is usually a consequence.
4. **Keep artifact names stable**
   Use deterministic file names (e.g., `/tmp/ref_<arch>.npz`, `/tmp/talu_<arch>.npz`)
   so different contributors can reproduce and compare quickly.
5. **Turn every root cause into a regression test**
   Add a focused test for config parsing, tensor layout/orientation, quant-dequant scaling,
   or routing edge cases depending on the failure class.

---

## Existing architectures

| Directory | Architecture | Model types | Heterogeneous |
|---|---|---|---|
| `llama/llama2.py` | llama2 | llama2, mistral, yi, vicuna, tinyllama | No |
| `llama/llama3.py` | llama3 | llama, llama3, llama3.1, llama3.2 | No |
| `qwen/qwen3.py` | qwen3 | qwen3, qwen3_vl, qwen2.5, qwen2, qwen | No |
| `qwen/qwen3_moe.py` | qwen3_moe | qwen3_moe | No |
| `qwen/qwen3_next.py` | qwen3_next | qwen3_next | Yes (linear_attention + full_attention) |
| `gemma/gemma3.py` | gemma3 | gemma3, gemma3_text, gemma2, gemma | No |
| `phi/phi4.py` | phi | phi3, phi4, phi | No |
| `granite/granite3.py` | granite3 | granite | No |
| `mistral/ministral3.py` | ministral3 | ministral3, mistral3 | No |
| `granite/granite_hybrid.py` | granite_hybrid | granite_hybrid, granitehybrid | Yes (mamba + attention) |
| `lfm2/lfm2.py` | lfm2 | lfm2 | Yes (shortconv + attention) |
| `bert/minilm.py` | minilm | bert, minilm | No |

## Debugging tips

- **Primitives mode**: `TALU_PRIMITIVES_ONLY=1 uv run python -m trace myarch`
  traces without fusing attention/mlp, useful for debugging non-standard
  architectures.
- **Weight mismatch**: if talu reports `MissingWeight`, compare your Block's
  `state_dict()` keys against the HuggingFace checkpoint to find naming
  differences.
- **Graph inspection**: read the generated `_graphs/myarch.json` -- each
  op should map to a real operation in your Block's forward pass.
