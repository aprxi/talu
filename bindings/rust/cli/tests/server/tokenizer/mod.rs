//! Integration tests for `/v1/tokenizer/*` endpoints.

mod api;

/// Minimal tokenizer fixture used by tokenizer server integration tests.
///
/// This avoids relying on external `tokenizer.json` files not present in all
/// test environments.
const TOKENIZER_JSON: &str = r##"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<PAD>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3,
      " ": 4,
      "b": 5,
      "d": 6,
      "e": 7,
      "g": 8,
      "h": 9,
      "l": 10,
      "o": 11,
      "r": 12,
      "w": 13,
      "y": 14
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<PAD>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null
}"##;

pub fn tokenizer_fixture_json() -> &'static str {
    TOKENIZER_JSON
}
