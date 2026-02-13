//! Shared test fixtures for the tokenizer integration test suite.
//!
//! Provides a minimal BPE tokenizer (99 tokens: 4 special + 95 ASCII printable)
//! that can be instantiated from JSON without model files.

use std::ffi::c_void;
use std::ptr;

// ---------------------------------------------------------------------------
// Correctly-typed FFI declarations
// ---------------------------------------------------------------------------
//
// The auto-generated talu_sys bindings declare several option parameters as
// by-value structs (e.g. `options: EncodeOptions`), but the Zig C API exports
// declare them as nullable pointers (`options: ?*const EncodeOptions`).
//
// Passing by value happens to work when all bytes are zero (interpreted as a
// null pointer on the Zig side), but crashes for non-default options because
// the struct bytes are read as a pointer address. These declarations use the
// correct pointer types so that non-default options work.
extern "C" {
    fn talu_tokenizer_encode(
        handle: *mut c_void,
        text: *const u8,
        text_len: usize,
        options: *const talu_sys::EncodeOptions,
    ) -> talu_sys::EncodeResult;

    fn talu_tokenizer_decode(
        handle: *mut c_void,
        tokens: *const u32,
        num_tokens: usize,
        options: *const talu_sys::DecodeOptionsC,
    ) -> talu_sys::DecodeResult;

    fn talu_tokenizer_encode_batch(
        handle: *mut c_void,
        texts: *const *const u8,
        lengths: *const usize,
        num_texts: usize,
        options: *const talu_sys::EncodeOptions,
    ) -> talu_sys::BatchEncodeResult;

    fn talu_batch_to_padded_tensor(
        ids: *const u32,
        offsets: *const usize,
        num_sequences: usize,
        options: *const talu_sys::PaddedTensorOptions,
    ) -> talu_sys::PaddedTensorResult;
}

/// Minimal BPE tokenizer JSON with 99 tokens.
///
/// Vocab layout:
///   0: `<pad>`, 1: `<s>`, 2: `</s>`, 3: `<unk>`,
///   4..=98: ASCII 0x20..=0x7E (space through tilde).
///
/// No merges, ByteLevel pre-tokenizer/decoder, no normalizer or post-processor.
///
/// Note: ByteLevel pre-tokenizer remaps space (0x20) to U+0120 (Ġ), which is
/// outside this vocab. Encode/decode roundtrips work for non-space ASCII.
pub const TOKENIZER_JSON: &str = r##"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      " ": 4, "!": 5, "\"": 6, "#": 7, "$": 8, "%": 9,
      "&": 10, "'": 11, "(": 12, ")": 13, "*": 14, "+": 15,
      ",": 16, "-": 17, ".": 18, "/": 19,
      "0": 20, "1": 21, "2": 22, "3": 23, "4": 24,
      "5": 25, "6": 26, "7": 27, "8": 28, "9": 29,
      ":": 30, ";": 31, "<": 32, "=": 33, ">": 34, "?": 35, "@": 36,
      "A": 37, "B": 38, "C": 39, "D": 40, "E": 41, "F": 42,
      "G": 43, "H": 44, "I": 45, "J": 46, "K": 47, "L": 48,
      "M": 49, "N": 50, "O": 51, "P": 52, "Q": 53, "R": 54,
      "S": 55, "T": 56, "U": 57, "V": 58, "W": 59, "X": 60,
      "Y": 61, "Z": 62, "[": 63, "\\": 64, "]": 65, "^": 66,
      "_": 67, "`": 68,
      "a": 69, "b": 70, "c": 71, "d": 72, "e": 73, "f": 74,
      "g": 75, "h": 76, "i": 77, "j": 78, "k": 79, "l": 80,
      "m": 81, "n": 82, "o": 83, "p": 84, "q": 85, "r": 86,
      "s": 87, "t": 88, "u": 89, "v": 90, "w": 91, "x": 92,
      "y": 93, "z": 94, "{": 95, "|": 96, "}": 97, "~": 98
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"##;

/// BPE tokenizer with merge rules (105 tokens: 99 base + 6 merged).
///
/// Extends the base fixture with merge rules so that BPE subword merging
/// actually occurs. This exercises the tokenizer's core merge algorithm.
///
/// Merged tokens (IDs 99–104):
///   99: "he", 100: "ll", 101: "lo", 102: "hel", 103: "hell", 104: "hello"
///
/// Merge rules (applied in rank order):
///   0: h+e→he, 1: he+l→hel, 2: hel+l→hell, 3: hell+o→hello,
///   4: l+l→ll, 5: l+o→lo
///
/// Expected tokenizations:
///   "hello" → [104]         (fully merged)
///   "hell"  → [103]         (merges 0–2)
///   "helo"  → [102, 83]    (hel + o; "hel o" is not a merge)
///   "llo"   → [100, 83]    (ll + o; "ll o" is not a merge)
///   "lo"    → [101]         (merge 5)
///   "hi"    → [76, 77]     (no merge for h+i)
///   "abc"   → [69, 70, 71] (no merges)
pub const MERGES_TOKENIZER_JSON: &str = r##"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
      " ": 4, "!": 5, "\"": 6, "#": 7, "$": 8, "%": 9,
      "&": 10, "'": 11, "(": 12, ")": 13, "*": 14, "+": 15,
      ",": 16, "-": 17, ".": 18, "/": 19,
      "0": 20, "1": 21, "2": 22, "3": 23, "4": 24,
      "5": 25, "6": 26, "7": 27, "8": 28, "9": 29,
      ":": 30, ";": 31, "<": 32, "=": 33, ">": 34, "?": 35, "@": 36,
      "A": 37, "B": 38, "C": 39, "D": 40, "E": 41, "F": 42,
      "G": 43, "H": 44, "I": 45, "J": 46, "K": 47, "L": 48,
      "M": 49, "N": 50, "O": 51, "P": 52, "Q": 53, "R": 54,
      "S": 55, "T": 56, "U": 57, "V": 58, "W": 59, "X": 60,
      "Y": 61, "Z": 62, "[": 63, "\\": 64, "]": 65, "^": 66,
      "_": 67, "`": 68,
      "a": 69, "b": 70, "c": 71, "d": 72, "e": 73, "f": 74,
      "g": 75, "h": 76, "i": 77, "j": 78, "k": 79, "l": 80,
      "m": 81, "n": 82, "o": 83, "p": 84, "q": 85, "r": 86,
      "s": 87, "t": 88, "u": 89, "v": 90, "w": 91, "x": 92,
      "y": 93, "z": 94, "{": 95, "|": 96, "}": 97, "~": 98,
      "he": 99, "ll": 100, "lo": 101, "hel": 102, "hell": 103, "hello": 104
    },
    "merges": [
      "h e",
      "he l",
      "hel l",
      "hell o",
      "l l",
      "l o"
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"##;

/// RAII wrapper for a tokenizer handle created from JSON.
///
/// Calls `talu_tokenizer_free` on drop.
pub struct TokenizerTestContext {
    handle: *mut c_void,
}

impl TokenizerTestContext {
    /// Create a tokenizer from the embedded minimal JSON fixture (no merges).
    pub fn new() -> Self {
        Self::from_json(TOKENIZER_JSON)
    }

    /// Create a tokenizer from the merges fixture (with BPE merge rules).
    pub fn with_merges() -> Self {
        Self::from_json(MERGES_TOKENIZER_JSON)
    }

    /// Create a tokenizer from the byte-level fixture (full 256 byte tokens).
    pub fn with_byte_level() -> Self {
        Self::from_json(&build_byte_level_tokenizer_json())
    }

    /// Create a tokenizer where special tokens are only in `added_tokens`
    /// (not in `model.vocab`), enabling observable `skip_special_tokens` behavior.
    pub fn with_special_tokens() -> Self {
        Self::from_json(SPECIAL_TOKENS_TOKENIZER_JSON)
    }

    /// Create a tokenizer from arbitrary JSON.
    pub fn from_json(json_str: &str) -> Self {
        let json = json_str.as_bytes();
        let mut handle: *mut c_void = ptr::null_mut();
        let rc = unsafe {
            talu_sys::talu_tokenizer_create_from_json(
                json.as_ptr(),
                json.len(),
                &mut handle as *mut _ as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "talu_tokenizer_create_from_json failed: {rc}");
        assert!(!handle.is_null(), "tokenizer handle is null after creation");
        Self { handle }
    }

    /// Raw tokenizer handle for passing to C API functions.
    pub fn handle(&self) -> *mut c_void {
        self.handle
    }

    /// Encode text and return owned token IDs (frees C-allocated buffer).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, &talu_sys::EncodeOptions::default())
    }

    /// Encode text with explicit options and return owned token IDs.
    pub fn encode_with(&self, text: &str, options: &talu_sys::EncodeOptions) -> Vec<u32> {
        let result = unsafe {
            talu_tokenizer_encode(
                self.handle,
                text.as_bytes().as_ptr(),
                text.len(),
                options as *const _,
            )
        };
        assert!(
            result.error_msg.is_null(),
            "encode failed: error_msg is non-null"
        );
        let tokens = if result.ids.is_null() || result.num_tokens == 0 {
            Vec::new()
        } else {
            let slice =
                unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
            let v = slice.to_vec();
            v
        };
        unsafe { talu_sys::talu_encode_result_free(result) };
        tokens
    }

    /// Decode token IDs back to text (frees C-allocated buffer).
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.decode_with(tokens, &talu_sys::DecodeOptionsC::default())
    }

    /// Decode token IDs with explicit options.
    pub fn decode_with(&self, tokens: &[u32], options: &talu_sys::DecodeOptionsC) -> String {
        let result = unsafe {
            talu_tokenizer_decode(
                self.handle,
                tokens.as_ptr(),
                tokens.len(),
                options as *const _,
            )
        };
        assert!(
            result.error_msg.is_null(),
            "decode failed: error_msg is non-null"
        );
        if result.text.is_null() || result.text_len == 0 {
            return String::new();
        }
        let text = unsafe {
            let slice = std::slice::from_raw_parts(result.text, result.text_len);
            String::from_utf8_lossy(slice).to_string()
        };
        unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
        text
    }

    /// Batch encode multiple texts and return (ids, offsets, num_sequences).
    pub fn encode_batch(
        &self,
        texts: &[&str],
        options: &talu_sys::EncodeOptions,
    ) -> BatchResult {
        let ptrs: Vec<*const u8> = texts.iter().map(|t| t.as_bytes().as_ptr()).collect();
        let lengths: Vec<usize> = texts.iter().map(|t| t.len()).collect();

        let result = unsafe {
            talu_tokenizer_encode_batch(
                self.handle,
                ptrs.as_ptr(),
                lengths.as_ptr(),
                texts.len(),
                options as *const _,
            )
        };
        assert!(
            result.error_msg.is_null(),
            "encode_batch failed: error_msg is non-null"
        );

        let ids = if result.ids.is_null() || result.total_tokens == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.ids, result.total_tokens) }.to_vec()
        };

        let offsets = if result.offsets.is_null() || result.num_sequences == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(result.offsets, result.num_sequences + 1) }.to_vec()
        };

        // Free C-allocated memory.
        if !result.ids.is_null() && result.total_tokens > 0 {
            unsafe {
                talu_sys::talu_batch_encode_result_free(
                    result.ids,
                    result.offsets,
                    result.total_tokens,
                    result.num_sequences,
                )
            };
        }

        BatchResult {
            ids,
            offsets,
            num_sequences: result.num_sequences,
        }
    }

    /// Batch encode then convert to padded tensor.
    pub fn batch_to_padded_tensor(
        &self,
        texts: &[&str],
        encode_opts: &talu_sys::EncodeOptions,
        pad_opts: &talu_sys::PaddedTensorOptions,
    ) -> PaddedTensorResult {
        let ptrs: Vec<*const u8> = texts.iter().map(|t| t.as_bytes().as_ptr()).collect();
        let lengths: Vec<usize> = texts.iter().map(|t| t.len()).collect();

        let batch = unsafe {
            talu_tokenizer_encode_batch(
                self.handle,
                ptrs.as_ptr(),
                lengths.as_ptr(),
                texts.len(),
                encode_opts as *const _,
            )
        };
        assert!(batch.error_msg.is_null(), "encode_batch failed");

        let tensor = unsafe {
            talu_batch_to_padded_tensor(
                batch.ids,
                batch.offsets,
                batch.num_sequences,
                pad_opts as *const _,
            )
        };
        assert!(tensor.error_msg.is_null(), "padded tensor failed");

        let total = tensor.num_sequences * tensor.padded_length;
        let input_ids = if tensor.input_ids.is_null() || total == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(tensor.input_ids, total) }.to_vec()
        };
        let attention_mask = if tensor.attention_mask.is_null() || total == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(tensor.attention_mask, total) }.to_vec()
        };

        // Free both results.
        unsafe {
            talu_sys::talu_padded_tensor_result_free(
                tensor.input_ids,
                tensor.attention_mask,
                tensor.num_sequences,
                tensor.padded_length,
            );
            if !batch.ids.is_null() && batch.total_tokens > 0 {
                talu_sys::talu_batch_encode_result_free(
                    batch.ids,
                    batch.offsets,
                    batch.total_tokens,
                    batch.num_sequences,
                );
            }
        }

        PaddedTensorResult {
            input_ids,
            attention_mask,
            num_sequences: tensor.num_sequences,
            padded_length: tensor.padded_length,
        }
    }
}

impl Drop for TokenizerTestContext {
    fn drop(&mut self) {
        unsafe { talu_sys::talu_tokenizer_free(self.handle) };
    }
}

/// Owned batch encoding result.
pub struct BatchResult {
    pub ids: Vec<u32>,
    pub offsets: Vec<usize>,
    pub num_sequences: usize,
}

/// Owned padded tensor result.
pub struct PaddedTensorResult {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub num_sequences: usize,
    pub padded_length: usize,
}

// ---------------------------------------------------------------------------
// Byte-level BPE fixture (full 256 byte tokens)
// ---------------------------------------------------------------------------

/// Compute the GPT-2 byte-to-unicode mapping table.
///
/// ByteLevel pre-tokenizer remaps each byte to a Unicode codepoint:
/// - Bytes 33–126 → same codepoint (printable ASCII, direct)
/// - Bytes 161–172 → same codepoint (Latin supplement, direct)
/// - Bytes 174–255 → same codepoint (Latin supplement, direct)
/// - Remaining 68 bytes (0–32, 127–160, 173) → codepoints 256–323 (shifted)
fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut shift = 256u32;

    for b in 0u16..=255 {
        let is_direct = (33..=126).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b);

        if is_direct {
            table[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            table[b as usize] = char::from_u32(shift).unwrap();
            shift += 1;
        }
    }
    table
}

/// Build a byte-level BPE tokenizer JSON with all 256 byte tokens.
///
/// Vocab: 260 tokens (4 special + 256 byte-level).
/// Every input byte maps to a real token (never `<unk>`), enabling
/// correct encode→decode roundtrips for any Unicode text.
///
/// Token ID layout: byte b → ID b + 4 (e.g., byte 0 → ID 4, byte 0x20 → ID 36).
pub fn build_byte_level_tokenizer_json() -> String {
    let table = gpt2_byte_to_unicode();
    let mut vocab_entries = Vec::with_capacity(260);

    // Special tokens
    vocab_entries.push(r#""<pad>": 0"#.to_string());
    vocab_entries.push(r#""<s>": 1"#.to_string());
    vocab_entries.push(r#""</s>": 2"#.to_string());
    vocab_entries.push(r#""<unk>": 3"#.to_string());

    // 256 byte-level tokens
    for b in 0u16..=255 {
        let ch = table[b as usize];
        let id = b + 4;
        // JSON-escape the character
        let key = match ch {
            '"' => r#"\""#.to_string(),
            '\\' => r#"\\"#.to_string(),
            c if c.is_ascii_graphic() || c == ' ' => c.to_string(),
            c => format!("\\u{:04X}", c as u32),
        };
        vocab_entries.push(format!("\"{key}\": {id}"));
    }

    format!(
        r#"{{
  "version": "1.0",
  "model": {{
    "type": "BPE",
    "vocab": {{
      {}
    }},
    "merges": []
  }},
  "added_tokens": [
    {{"id": 0, "content": "<pad>", "special": true}},
    {{"id": 1, "content": "<s>", "special": true}},
    {{"id": 2, "content": "</s>", "special": true}},
    {{"id": 3, "content": "<unk>", "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel"}}
}}"#,
        vocab_entries.join(",\n      ")
    )
}

/// Token ID for a given byte value in the byte-level fixture.
///
/// Each byte b maps to token ID b + 4.
pub fn byte_token_id(byte: u8) -> u32 {
    byte as u32 + 4
}

// ---------------------------------------------------------------------------
// Post-processor fixture (adds BOS/EOS during encoding)
// ---------------------------------------------------------------------------

/// Same vocab as `TOKENIZER_JSON` but special tokens (IDs 0–3) are ONLY in
/// `added_tokens`, not in `model.vocab`. This makes `skip_special_tokens`
/// observable during decode: the BPE decoder checks `added_tokens` for IDs
/// not in the vocab and correctly reads their `special` flag.
///
/// Regular tokens start at ID 4 (" ") through 98 ("~"), same as the base
/// fixture shifted by 0 (IDs are unchanged).
pub const SPECIAL_TOKENS_TOKENIZER_JSON: &str = r##"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      " ": 4, "!": 5, "\"": 6, "#": 7, "$": 8, "%": 9,
      "&": 10, "'": 11, "(": 12, ")": 13, "*": 14, "+": 15,
      ",": 16, "-": 17, ".": 18, "/": 19,
      "0": 20, "1": 21, "2": 22, "3": 23, "4": 24,
      "5": 25, "6": 26, "7": 27, "8": 28, "9": 29,
      ":": 30, ";": 31, "<": 32, "=": 33, ">": 34, "?": 35, "@": 36,
      "A": 37, "B": 38, "C": 39, "D": 40, "E": 41, "F": 42,
      "G": 43, "H": 44, "I": 45, "J": 46, "K": 47, "L": 48,
      "M": 49, "N": 50, "O": 51, "P": 52, "Q": 53, "R": 54,
      "S": 55, "T": 56, "U": 57, "V": 58, "W": 59, "X": 60,
      "Y": 61, "Z": 62, "[": 63, "\\": 64, "]": 65, "^": 66,
      "_": 67, "`": 68,
      "a": 69, "b": 70, "c": 71, "d": 72, "e": 73, "f": 74,
      "g": 75, "h": 76, "i": 77, "j": 78, "k": 79, "l": 80,
      "m": 81, "n": 82, "o": 83, "p": 84, "q": 85, "r": 86,
      "s": 87, "t": 88, "u": 89, "v": 90, "w": 91, "x": 92,
      "y": 93, "z": 94, "{": 95, "|": 96, "}": 97, "~": 98
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"##;

// ---------------------------------------------------------------------------
// Raw FFI wrappers for stress tests that share a handle across threads
// ---------------------------------------------------------------------------

/// Encode text via the correctly-typed FFI, returning the raw result.
///
/// # Safety
/// `handle` must be a valid tokenizer handle.
pub unsafe fn encode_raw(
    handle: *mut c_void,
    text: &[u8],
    options: &talu_sys::EncodeOptions,
) -> talu_sys::EncodeResult {
    talu_tokenizer_encode(handle, text.as_ptr(), text.len(), options as *const _)
}

/// Decode tokens via the correctly-typed FFI, returning the raw result.
///
/// # Safety
/// `handle` must be a valid tokenizer handle.
pub unsafe fn decode_raw(
    handle: *mut c_void,
    tokens: &[u32],
    options: &talu_sys::DecodeOptionsC,
) -> talu_sys::DecodeResult {
    talu_tokenizer_decode(handle, tokens.as_ptr(), tokens.len(), options as *const _)
}
