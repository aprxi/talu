# talu

Safe, idiomatic Rust SDK for [talu](https://github.com/aprxi/talu) LLM inference.

Talu is a local-first inference engine built from scratch in Zig. This crate
provides RAII-based resource management and `Result`-based error handling over
the raw FFI layer in [`talu-sys`](https://crates.io/crates/talu-sys).

## Features

- **Chat** — multi-turn conversation with streaming generation
- **Inference backends** — load and run local models (CPU, Metal)
- **Tokenizer** — encode and decode text
- **Embeddings & vector search** — embed text and search over vectors
- **Document storage** — store, search, and manage documents
- **Structured output** — constrained generation with grammar validation
- **Model management** — download, list, and inspect cached models
- **Responses API** — item-based conversation inspection (Open Responses compatible)

## Example

```rust
use talu::{ChatHandle, InferenceBackend};

let chat = ChatHandle::new(Some("You are a helpful assistant."))?;
let backend = InferenceBackend::new("path/to/model")?;
// Use chat and backend for generation...
# Ok::<(), talu::Error>(())
```

### Responses API

```rust
use talu::responses::{ResponsesHandle, ResponsesView, MessageRole, ItemType};

let mut conv = ResponsesHandle::new()?;
conv.append_message(MessageRole::User, "Hello!")?;

for item in conv.items() {
    let item = item?;
    match item.item_type {
        ItemType::Message => {
            let msg = conv.get_message(0)?;
            println!("Role: {:?}", msg.role);
        }
        _ => {}
    }
}
# Ok::<(), talu::Error>(())
```

## Requirements

The `talu-sys` dependency automatically downloads the pre-built native library
at build time. See the [`talu-sys` README](https://crates.io/crates/talu-sys)
for details on supported targets and custom builds.

## License

MIT
