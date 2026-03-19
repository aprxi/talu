//! Stateful `/v1/tokenizer/*` HTTP endpoints.
//!
//! Provides tokenizer instance lifecycle and encode/decode operations backed by
//! core tokenizer C APIs.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    Tokenizers,
    Talu,
}

impl TokenizerBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tokenizers => "tokenizers",
            Self::Talu => "talu",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "tokenizers" => Some(Self::Tokenizers),
            "talu" => Some(Self::Talu),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstanceTruncationConfig {
    pub max_length: usize,
    pub direction: TruncationDirection,
}

#[derive(Debug, Clone, Copy)]
pub enum TruncationDirection {
    Right,
    Left,
}

impl TruncationDirection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Right => "right",
            Self::Left => "left",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstancePaddingConfig {
    pub direction: PaddingDirection,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
    pub length: Option<usize>,
    pub multiple_of: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum PaddingDirection {
    Right,
    Left,
}

impl PaddingDirection {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "right" => Some(Self::Right),
            "left" => Some(Self::Left),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Right => "right",
            Self::Left => "left",
        }
    }
}

#[derive(Debug)]
pub struct TokenizerOwnedHandle {
    ptr: *mut c_void,
}

impl TokenizerOwnedHandle {
    fn from_raw(ptr: *mut c_void) -> Self {
        Self { ptr }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for TokenizerOwnedHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: pointer was returned by talu_tokenizer_create*.
            unsafe { talu_sys::talu_tokenizer_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Raw pointers are used behind AppState mutex for C API calls.
unsafe impl Send for TokenizerOwnedHandle {}
unsafe impl Sync for TokenizerOwnedHandle {}

#[derive(Debug)]
pub struct TokenizerInstance {
    pub backend: TokenizerBackend,
    pub source_kind: String,
    pub source_value: String,
    pub tokenizer_sha256: String,
    pub vocab_size: usize,
    pub handle: TokenizerOwnedHandle,
    pub truncation: Option<InstanceTruncationConfig>,
    pub padding: Option<InstancePaddingConfig>,
    pub added_tokens: Vec<AddedTokenEntry>,
}

#[derive(Debug, Clone)]
pub struct AddedTokenEntry {
    pub content: String,
    pub id: u32,
    pub special: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
}

#[derive(Debug, Deserialize)]
struct CreateInstanceRequest {
    backend: String,
    source: SourceRequest,
}

#[derive(Debug, Deserialize, Serialize)]
struct SourceRequest {
    kind: String,
    value: String,
}

#[derive(Debug, Serialize)]
struct CreateInstanceResponse {
    tokenizer_id: String,
    backend: String,
    tokenizer_sha256: String,
    vocab_size: usize,
}

#[derive(Debug, Serialize)]
struct InstanceResponse {
    tokenizer_id: String,
    backend: String,
    source: SourceRequest,
    tokenizer_sha256: String,
    vocab_size: usize,
    truncation: Option<TruncationResponse>,
    padding: Option<PaddingResponse>,
}

#[derive(Debug, Serialize)]
struct TruncationResponse {
    max_length: usize,
    direction: String,
}

#[derive(Debug, Serialize)]
struct PaddingResponse {
    direction: String,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
    length: Option<usize>,
    multiple_of: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TokenizerIdRequest {
    tokenizer_id: String,
}

#[derive(Debug, Deserialize)]
struct EncodeRequest {
    tokenizer_id: String,
    sequence: Value,
    #[serde(default)]
    pair: Option<Value>,
    #[serde(default)]
    is_pretokenized: Option<bool>,
    #[serde(default)]
    add_special_tokens: Option<bool>,
    #[serde(default, rename = "return")]
    return_fields: Option<ReturnFields>,
    #[serde(default)]
    benchmark: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct EncodeBatchRequest {
    tokenizer_id: String,
    inputs: Vec<Value>,
    #[serde(default)]
    is_pretokenized: Option<bool>,
    #[serde(default)]
    add_special_tokens: Option<bool>,
    #[serde(default)]
    padding: Option<PaddingRequest>,
    #[serde(default, rename = "return")]
    return_fields: Option<ReturnFields>,
    #[serde(default)]
    benchmark: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct PaddingRequest {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default, alias = "pad_token_id")]
    pad_id: Option<u32>,
    #[serde(default)]
    pad_type_id: Option<u32>,
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    length: Option<usize>,
    #[serde(default)]
    multiple_of: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ReturnFields {
    #[serde(default)]
    ids: Option<bool>,
    #[serde(default)]
    tokens: Option<bool>,
    #[serde(default)]
    type_ids: Option<bool>,
    #[serde(default)]
    attention_mask: Option<bool>,
    #[serde(default)]
    special_tokens_mask: Option<bool>,
    #[serde(default)]
    offsets: Option<bool>,
}

impl ReturnFields {
    fn include_ids(&self) -> bool {
        self.ids.unwrap_or(true)
    }
    fn include_tokens(&self) -> bool {
        self.tokens.unwrap_or(true)
    }
    fn include_type_ids(&self) -> bool {
        self.type_ids.unwrap_or(true)
    }
    fn include_attention_mask(&self) -> bool {
        self.attention_mask.unwrap_or(true)
    }
    fn include_special_tokens_mask(&self) -> bool {
        self.special_tokens_mask.unwrap_or(true)
    }
    fn include_offsets(&self) -> bool {
        self.offsets.unwrap_or(true)
    }
}

impl Default for ReturnFields {
    fn default() -> Self {
        Self {
            ids: Some(true),
            tokens: Some(true),
            type_ids: Some(true),
            attention_mask: Some(true),
            special_tokens_mask: Some(true),
            offsets: Some(true),
        }
    }
}

#[derive(Debug, Deserialize)]
struct DecodeRequest {
    tokenizer_id: String,
    ids: Vec<u32>,
    #[serde(default)]
    skip_special_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct DecodeBatchRequest {
    tokenizer_id: String,
    ids_batch: Vec<Vec<u32>>,
    #[serde(default)]
    skip_special_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TokenToIdRequest {
    tokenizer_id: String,
    token: String,
}

#[derive(Debug, Deserialize)]
struct IdToTokenRequest {
    tokenizer_id: String,
    token_id: i32,
}

#[derive(Debug, Deserialize)]
struct AddTokensRequest {
    tokenizer_id: String,
    tokens: Vec<AddedTokenRequest>,
}

#[derive(Debug, Deserialize)]
struct AddSpecialTokensRequest {
    tokenizer_id: String,
    #[serde(default)]
    tokens: Option<Vec<AddedTokenRequest>>,
    #[serde(default)]
    special_tokens: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AddedTokenRequest {
    Text(String),
    Object(AddedTokenObjectRequest),
}

#[derive(Debug, Deserialize)]
struct AddedTokenObjectRequest {
    content: String,
    #[serde(default)]
    single_word: Option<bool>,
    #[serde(default)]
    lstrip: Option<bool>,
    #[serde(default)]
    rstrip: Option<bool>,
    #[serde(default)]
    normalized: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SaveRequest {
    tokenizer_id: String,
    path: String,
    #[serde(default)]
    pretty: Option<bool>,
    #[serde(default)]
    overwrite: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TrainRequest {
    tokenizer_id: String,
    #[serde(default)]
    texts: Option<Vec<String>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    trainer: Option<TrainerRequest>,
}

#[derive(Debug, Deserialize)]
struct TrainFromIteratorRequest {
    tokenizer_id: String,
    #[serde(default)]
    iterator: Option<Vec<String>>,
    #[serde(default)]
    texts: Option<Vec<String>>,
    #[serde(default)]
    trainer: Option<TrainerRequest>,
}

#[derive(Debug, Deserialize)]
struct TrainerRequest {
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    special_tokens: Option<Value>,
}

#[derive(Debug)]
struct ParsedAddedToken {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

#[derive(Debug, Deserialize)]
struct EnableTruncationRequest {
    tokenizer_id: String,
    max_length: usize,
    #[serde(default)]
    stride: Option<usize>,
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    direction: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EnablePaddingRequest {
    tokenizer_id: String,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default, alias = "pad_token_id")]
    pad_id: Option<u32>,
    #[serde(default)]
    pad_type_id: Option<u32>,
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    length: Option<usize>,
    #[serde(default)]
    multiple_of: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CompareRequest {
    left_tokenizer_id: String,
    right_tokenizer_id: String,
    sequence: String,
    #[serde(default)]
    add_special_tokens: Option<bool>,
    #[serde(default)]
    window: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TimingMs {
    encode: f64,
    total: f64,
}

#[derive(Debug, Serialize, Clone)]
struct EncodingPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    type_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_mask: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_tokens_mask: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offsets: Option<Vec<[u32; 2]>>,
}

#[derive(Debug, Serialize)]
struct EncodeResponse {
    encoding: EncodingPayload,
    sha256_ids: String,
    #[serde(rename = "impl")]
    impl_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_ms: Option<TimingMs>,
}

#[derive(Debug, Serialize)]
struct BatchEncodingPayload {
    encodings: Vec<EncodingPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_ids: Option<Vec<Vec<u32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_mask: Option<Vec<Vec<u32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    type_ids: Option<Vec<Vec<u32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_tokens_mask: Option<Vec<Vec<u32>>>,
    lengths: Vec<usize>,
    num_sequences: usize,
    total_tokens: usize,
    padding_side: String,
    pad_token_id: u32,
}

#[derive(Debug, Serialize)]
struct EncodeBatchResponse {
    batch_encoding: BatchEncodingPayload,
    sha256_ids_batch: Vec<String>,
    #[serde(rename = "impl")]
    impl_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_ms: Option<TimingMs>,
}

#[derive(Debug, Clone)]
struct MutableEncoding {
    ids: Vec<u32>,
    tokens: Vec<String>,
    type_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    special_tokens_mask: Vec<u32>,
    offsets: Vec<[u32; 2]>,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

// ---------------------------------------------------------------------------
// Instance lifecycle
// ---------------------------------------------------------------------------

pub async fn handle_create_instance(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: CreateInstanceRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let backend = match TokenizerBackend::parse(request.backend.as_str()) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "unsupported_backend",
                "backend must be one of: tokenizers, talu",
                Some(json!({ "field": "backend" })),
            )
        }
    };

    let source_kind = request.source.kind.as_str();
    let source_value = request.source.value;

    if source_kind != "path" && source_kind != "json" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "source.kind must be one of: path, json",
            Some(json!({ "field": "source.kind" })),
        );
    }

    let (tokenizer_id, tokenizer_sha256, vocab_size, instance) = {
        let mut handle_ptr: *mut c_void = std::ptr::null_mut();
        let create_rc = if source_kind == "path" {
            let c_path = match CString::new(source_value.clone()) {
                Ok(v) => v,
                Err(_) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "source.value contains interior NUL bytes",
                        Some(json!({ "field": "source.value" })),
                    )
                }
            };
            // SAFETY: c_path is NUL-terminated and output pointer is valid.
            unsafe {
                talu_sys::talu_tokenizer_create(
                    c_path.as_ptr(),
                    &mut handle_ptr as *mut _ as *mut c_void,
                )
            }
        } else {
            // SAFETY: source_value bytes are valid for the duration of the call.
            unsafe {
                talu_sys::talu_tokenizer_create_from_json(
                    source_value.as_bytes().as_ptr(),
                    source_value.len(),
                    &mut handle_ptr as *mut _ as *mut c_void,
                )
            }
        };

        if create_rc != 0 || handle_ptr.is_null() {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!(
                    "failed to create tokenizer instance: {}",
                    take_last_error().unwrap_or_else(|| "unknown error".to_string())
                ),
                None,
            );
        }

        let tokenizer_sha256 = if source_kind == "path" {
            tokenizer_sha256_from_path_source(&source_value, handle_ptr)
        } else {
            sha256_hex(source_value.as_bytes())
        };

        // SAFETY: handle is valid from create call above.
        let vocab_size = unsafe { talu_sys::talu_tokenizer_get_vocab_size(handle_ptr) };

        let tokenizer_id = format!("tok_{}", Uuid::new_v4().simple());
        let instance = TokenizerInstance {
            backend,
            source_kind: source_kind.to_string(),
            source_value: source_value.clone(),
            tokenizer_sha256: tokenizer_sha256.clone(),
            vocab_size,
            handle: TokenizerOwnedHandle::from_raw(handle_ptr),
            truncation: None,
            padding: None,
            added_tokens: Vec::new(),
        };

        (tokenizer_id, tokenizer_sha256, vocab_size, instance)
    };

    let mut instances = state.tokenizer_instances.lock().await;
    instances.insert(tokenizer_id.clone(), instance);

    json_response(
        StatusCode::OK,
        &CreateInstanceResponse {
            tokenizer_id,
            backend: backend.as_str().to_string(),
            tokenizer_sha256,
            vocab_size,
        },
    )
}

pub async fn handle_get_instance(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let tokenizer_id = match extract_instance_id(req.uri().path()) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "missing tokenizer_id in path",
                None,
            )
        }
    };

    let instances = state.tokenizer_instances.lock().await;
    let Some(instance) = instances.get(tokenizer_id) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "tokenizer_not_found",
            "tokenizer instance not found",
            Some(json!({ "tokenizer_id": tokenizer_id })),
        );
    };

    let truncation = instance.truncation.as_ref().map(|t| TruncationResponse {
        max_length: t.max_length,
        direction: t.direction.as_str().to_string(),
    });
    let padding = instance.padding.as_ref().map(|p| PaddingResponse {
        direction: p.direction.as_str().to_string(),
        pad_id: p.pad_id,
        pad_type_id: p.pad_type_id,
        pad_token: p.pad_token.clone(),
        length: p.length,
        multiple_of: p.multiple_of,
    });

    json_response(
        StatusCode::OK,
        &InstanceResponse {
            tokenizer_id: tokenizer_id.to_string(),
            backend: instance.backend.as_str().to_string(),
            source: SourceRequest {
                kind: instance.source_kind.clone(),
                value: instance.source_value.clone(),
            },
            tokenizer_sha256: instance.tokenizer_sha256.clone(),
            vocab_size: instance.vocab_size,
            truncation,
            padding,
        },
    )
}

pub async fn handle_delete_instance(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let tokenizer_id = match extract_instance_id(req.uri().path()) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "missing tokenizer_id in path",
                None,
            )
        }
    };

    let mut instances = state.tokenizer_instances.lock().await;
    if instances.remove(tokenizer_id).is_none() {
        return json_error(
            StatusCode::NOT_FOUND,
            "tokenizer_not_found",
            "tokenizer instance not found",
            Some(json!({ "tokenizer_id": tokenizer_id })),
        );
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Encode/decode
// ---------------------------------------------------------------------------

pub async fn handle_encode(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: EncodeRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.pair.as_ref().is_some_and(|v| !v.is_null()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unsupported_option",
            "pair encoding is not supported by /v1/tokenizer/encode in this build",
            Some(json!({ "field": "pair" })),
        );
    }

    if request.is_pretokenized.unwrap_or(false) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unsupported_option",
            "is_pretokenized=true is not supported",
            Some(json!({ "field": "is_pretokenized" })),
        );
    }

    let sequence = match normalize_input_sequence(&request.sequence) {
        Ok(s) => s,
        Err(msg) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &msg,
                Some(json!({ "field": "sequence" })),
            )
        }
    };

    let add_special_tokens = request.add_special_tokens.unwrap_or(true);
    let return_fields = request.return_fields.unwrap_or_default();
    let include_timing = request.benchmark.unwrap_or(false);

    let total_start = Instant::now();

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let encode_start = Instant::now();
    let mut encoding = match encode_one(instance, &sequence, add_special_tokens) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "encode_failed", &msg, None),
    };

    if let Some(config) = instance.padding.clone() {
        apply_padding_to_encoding(&mut encoding, &config);
    }

    let encode_ms = elapsed_ms(encode_start);
    let total_ms = elapsed_ms(total_start);
    let payload = encoding_payload_from_mutable(&encoding, &return_fields);
    let sha256_ids = sha256_ids(&encoding.ids);

    json_response(
        StatusCode::OK,
        &EncodeResponse {
            encoding: payload,
            sha256_ids,
            impl_name: format!("{}.encode", instance.backend.as_str()),
            timing_ms: include_timing.then_some(TimingMs {
                encode: encode_ms,
                total: total_ms,
            }),
        },
    )
}

pub async fn handle_encode_batch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: EncodeBatchRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.is_pretokenized.unwrap_or(false) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unsupported_option",
            "is_pretokenized=true is not supported",
            Some(json!({ "field": "is_pretokenized" })),
        );
    }

    let inputs = match normalize_batch_inputs(&request.inputs) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &msg,
                Some(json!({ "field": "inputs" })),
            )
        }
    };

    let add_special_tokens = request.add_special_tokens.unwrap_or(true);
    let return_fields = request.return_fields.unwrap_or_default();
    let include_timing = request.benchmark.unwrap_or(false);

    let total_start = Instant::now();

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let effective_padding = match build_effective_padding(instance, request.padding) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let encode_start = Instant::now();
    let mut rows = Vec::with_capacity(inputs.len());
    for text in &inputs {
        let encoding = match encode_one(instance, text, add_special_tokens) {
            Ok(v) => v,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "encode_failed", &msg, None),
        };
        rows.push(encoding);
    }

    // We need mutable rows for padding and payload output.
    let mut mutable_rows: Vec<MutableEncoding> = rows;

    let lengths: Vec<usize> = mutable_rows
        .iter()
        .map(|row| row.attention_mask.iter().filter(|&&v| v != 0).count())
        .collect();

    if let Some(config) = effective_padding.as_ref() {
        apply_padding_to_batch(&mut mutable_rows, config);
    }

    let encode_ms = elapsed_ms(encode_start);
    let total_ms = elapsed_ms(total_start);

    let encodings: Vec<EncodingPayload> = mutable_rows
        .iter()
        .map(|row| encoding_payload_from_mutable(row, &return_fields))
        .collect();

    let input_ids = return_fields.include_ids().then(|| {
        mutable_rows
            .iter()
            .map(|row| row.ids.clone())
            .collect::<Vec<_>>()
    });
    let attention_mask = return_fields.include_attention_mask().then(|| {
        mutable_rows
            .iter()
            .map(|row| row.attention_mask.clone())
            .collect::<Vec<_>>()
    });
    let type_ids = return_fields.include_type_ids().then(|| {
        mutable_rows
            .iter()
            .map(|row| row.type_ids.clone())
            .collect::<Vec<_>>()
    });
    let special_tokens_mask = return_fields.include_special_tokens_mask().then(|| {
        mutable_rows
            .iter()
            .map(|row| row.special_tokens_mask.clone())
            .collect::<Vec<_>>()
    });

    let total_tokens = lengths.iter().sum();
    let sha256_ids_batch: Vec<String> = mutable_rows
        .iter()
        .map(|row| sha256_ids(&row.ids))
        .collect();

    let (padding_side, pad_token_id) = if let Some(config) = effective_padding {
        (config.direction.as_str().to_string(), config.pad_id)
    } else {
        ("right".to_string(), 0)
    };

    json_response(
        StatusCode::OK,
        &EncodeBatchResponse {
            batch_encoding: BatchEncodingPayload {
                encodings,
                input_ids,
                attention_mask,
                type_ids,
                special_tokens_mask,
                lengths,
                num_sequences: mutable_rows.len(),
                total_tokens,
                padding_side,
                pad_token_id,
            },
            sha256_ids_batch,
            impl_name: format!("{}.encode_batch", instance.backend.as_str()),
            timing_ms: include_timing.then_some(TimingMs {
                encode: encode_ms,
                total: total_ms,
            }),
        },
    )
}

pub async fn handle_decode(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: DecodeRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let skip_special_tokens = request.skip_special_tokens.unwrap_or(false);

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let text = match decode_ids_with_added_fallback(instance, &request.ids, skip_special_tokens) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "decode_failed", &msg, None),
    };

    json_response(StatusCode::OK, &json!({ "text": text }))
}

pub async fn handle_decode_batch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: DecodeBatchRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let skip_special_tokens = request.skip_special_tokens.unwrap_or(false);

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let mut texts = Vec::with_capacity(request.ids_batch.len());
    for ids in &request.ids_batch {
        let text = match decode_ids_with_added_fallback(instance, ids, skip_special_tokens) {
            Ok(v) => v,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "decode_failed", &msg, None),
        };
        texts.push(text);
    }

    json_response(StatusCode::OK, &json!({ "texts": texts }))
}

// ---------------------------------------------------------------------------
// Vocabulary and token mapping
// ---------------------------------------------------------------------------

pub async fn handle_vocab(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let query = parse_query(req.uri().query());
    let tokenizer_id = match query.get("tokenizer_id") {
        Some(v) if !v.is_empty() => v,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "missing query parameter tokenizer_id",
                Some(json!({ "field": "tokenizer_id" })),
            )
        }
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": tokenizer_id })),
            )
        }
    };

    // SAFETY: tokenizer handle is valid while instance is held.
    let vocab = unsafe { talu_sys::talu_tokenizer_get_vocab(instance.handle.as_ptr()) };
    if !vocab.error_msg.is_null() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &ffi_message(vocab.error_msg, "failed to fetch vocab"),
            None,
        );
    }

    let mut map = serde_json::Map::new();
    if vocab.num_entries > 0 {
        for idx in 0..vocab.num_entries {
            // SAFETY: returned arrays have num_entries elements per C API contract.
            let token_ptr = unsafe { *vocab.tokens.add(idx) };
            // SAFETY: lengths pointer valid for num_entries.
            let token_len = unsafe { *vocab.lengths.add(idx) as usize };
            // SAFETY: ids pointer valid for num_entries.
            let token_id = unsafe { *vocab.ids.add(idx) };
            // SAFETY: token_ptr points to token_len bytes; tokens can contain any UTF-8 bytes.
            let bytes = unsafe { std::slice::from_raw_parts(token_ptr as *const u8, token_len) };
            let token = String::from_utf8_lossy(bytes).to_string();
            map.insert(token, json!(token_id));
        }
    }

    for added in &instance.added_tokens {
        map.insert(added.content.clone(), json!(added.id));
    }

    // SAFETY: frees buffers returned by talu_tokenizer_get_vocab.
    unsafe {
        talu_sys::talu_vocab_result_free(vocab.tokens, vocab.lengths, vocab.ids, vocab.num_entries)
    };

    json_response(StatusCode::OK, &Value::Object(map))
}

pub async fn handle_vocab_size(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let query = parse_query(req.uri().query());
    let tokenizer_id = match query.get("tokenizer_id") {
        Some(v) if !v.is_empty() => v,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "missing query parameter tokenizer_id",
                Some(json!({ "field": "tokenizer_id" })),
            )
        }
    };

    let with_added_tokens = query
        .get("with_added_tokens")
        .map(|v| v == "true")
        .unwrap_or(true);

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": tokenizer_id })),
            )
        }
    };

    // SAFETY: tokenizer handle is valid while instance is held.
    let base_vocab_size =
        unsafe { talu_sys::talu_tokenizer_get_vocab_size(instance.handle.as_ptr()) };
    let vocab_size = if with_added_tokens {
        base_vocab_size.saturating_add(instance.added_tokens.len())
    } else {
        base_vocab_size
    };

    json_response(
        StatusCode::OK,
        &json!({
            "vocab_size": vocab_size,
            "with_added_tokens": with_added_tokens
        }),
    )
}

pub async fn handle_token_to_id(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: TokenToIdRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    if let Some(added) = instance
        .added_tokens
        .iter()
        .find(|added| added.content == request.token)
    {
        return json_response(StatusCode::OK, &json!({ "id": added.id }));
    }

    // SAFETY: handle is valid; token bytes pointer valid for call duration.
    let id = unsafe {
        talu_sys::talu_tokenizer_token_to_id(
            instance.handle.as_ptr(),
            request.token.as_bytes().as_ptr(),
            request.token.len(),
        )
    };

    if let Some(err) = take_last_error() {
        return json_error(StatusCode::BAD_REQUEST, "invalid_request", &err, None);
    }

    json_response(StatusCode::OK, &json!({ "id": id }))
}

pub async fn handle_id_to_token(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: IdToTokenRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let token = match id_to_token_with_added(instance, request.token_id) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", &msg, None),
    };

    json_response(StatusCode::OK, &json!({ "token": token }))
}

// ---------------------------------------------------------------------------
// Mutable configuration
// ---------------------------------------------------------------------------

pub async fn handle_enable_truncation(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: EnableTruncationRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.max_length == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "max_length must be greater than zero",
            Some(json!({ "field": "max_length" })),
        );
    }

    if request.stride.unwrap_or(0) != 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unsupported_option",
            "stride is not supported by this tokenizer backend",
            Some(json!({ "field": "stride" })),
        );
    }

    if let Some(strategy) = request.strategy.as_deref() {
        if strategy != "longest_first" {
            return json_error(
                StatusCode::BAD_REQUEST,
                "unsupported_option",
                "only strategy=longest_first is supported",
                Some(json!({ "field": "strategy" })),
            );
        }
    }

    let direction = match request.direction.as_deref().unwrap_or("right") {
        "right" => TruncationDirection::Right,
        "left" => TruncationDirection::Left,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "direction must be one of: left, right",
                Some(json!({ "field": "direction" })),
            )
        }
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    instance.truncation = Some(InstanceTruncationConfig {
        max_length: request.max_length,
        direction,
    });

    json_response(
        StatusCode::OK,
        &json!({
            "ok": true,
            "truncation": {
                "max_length": request.max_length,
                "direction": direction.as_str()
            }
        }),
    )
}

pub async fn handle_disable_truncation(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: TokenizerIdRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    instance.truncation = None;
    json_response(StatusCode::OK, &json!({ "ok": true }))
}

pub async fn handle_enable_padding(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: EnablePaddingRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.multiple_of == Some(0) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "multiple_of must be greater than zero",
            Some(json!({ "field": "multiple_of" })),
        );
    }

    let direction = match request.direction.as_deref().unwrap_or("right") {
        "right" => PaddingDirection::Right,
        "left" => PaddingDirection::Left,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "direction must be one of: left, right",
                Some(json!({ "field": "direction" })),
            )
        }
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let special = special_tokens(instance);
    let pad_id = request
        .pad_id
        .or_else(|| u32::try_from(special.pad_token_id).ok())
        .unwrap_or(0);
    let pad_token = match request.pad_token {
        Some(v) => v,
        None => id_to_token(instance, i32::try_from(pad_id).unwrap_or(-1))
            .unwrap_or_else(|_| "<PAD>".to_string()),
    };

    instance.padding = Some(InstancePaddingConfig {
        direction,
        pad_id,
        pad_type_id: request.pad_type_id.unwrap_or(0),
        pad_token: pad_token.clone(),
        length: request.length,
        multiple_of: request.multiple_of,
    });

    json_response(
        StatusCode::OK,
        &json!({
            "ok": true,
            "padding": {
                "direction": direction.as_str(),
                "pad_id": pad_id,
                "pad_type_id": request.pad_type_id.unwrap_or(0),
                "pad_token": pad_token,
                "length": request.length,
                "multiple_of": request.multiple_of
            }
        }),
    )
}

pub async fn handle_disable_padding(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: TokenizerIdRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    instance.padding = None;
    json_response(StatusCode::OK, &json!({ "ok": true }))
}

// ---------------------------------------------------------------------------
// Token mutation + serialization
// ---------------------------------------------------------------------------

pub async fn handle_add_tokens(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: AddTokensRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.tokens.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "tokens must be a non-empty array",
            Some(json!({ "field": "tokens" })),
        );
    }

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let mut added = 0usize;
    for token_req in request.tokens {
        let parsed = match parse_added_token_request(token_req, false) {
            Ok(v) => v,
            Err(msg) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    &msg,
                    Some(json!({ "field": "tokens" })),
                )
            }
        };

        if token_exists(instance, &parsed.content) {
            continue;
        }

        let id = next_added_token_id(instance);
        instance.added_tokens.push(AddedTokenEntry {
            content: parsed.content,
            id,
            special: false,
            single_word: parsed.single_word,
            lstrip: parsed.lstrip,
            rstrip: parsed.rstrip,
            normalized: parsed.normalized,
        });
        added += 1;
    }

    json_response(
        StatusCode::OK,
        &json!({
            "added": added,
            "vocab_size": effective_vocab_size(instance)
        }),
    )
}

pub async fn handle_add_special_tokens(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: AddSpecialTokensRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut parsed_tokens = Vec::new();
    if let Some(tokens) = request.tokens {
        for token in tokens {
            match parse_added_token_request(token, true) {
                Ok(v) => parsed_tokens.push(v),
                Err(msg) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        &msg,
                        Some(json!({ "field": "tokens" })),
                    )
                }
            }
        }
    }

    if let Some(special_tokens) = request.special_tokens {
        match extract_special_tokens_from_value(&special_tokens) {
            Ok(mut values) => parsed_tokens.append(&mut values),
            Err(msg) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    &msg,
                    Some(json!({ "field": "special_tokens" })),
                )
            }
        }
    }

    if parsed_tokens.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "expected tokens or special_tokens with at least one token",
            None,
        );
    }

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let mut added = 0usize;
    for parsed in parsed_tokens {
        if token_exists(instance, &parsed.content) {
            continue;
        }
        let id = next_added_token_id(instance);
        instance.added_tokens.push(AddedTokenEntry {
            content: parsed.content,
            id,
            special: true,
            single_word: parsed.single_word,
            lstrip: parsed.lstrip,
            rstrip: parsed.rstrip,
            normalized: parsed.normalized,
        });
        added += 1;
    }

    json_response(
        StatusCode::OK,
        &json!({
            "added": added,
            "vocab_size": effective_vocab_size(instance)
        }),
    )
}

pub async fn handle_train(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: TrainRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let texts = match collect_train_texts(request.texts, request.text) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", &msg, None),
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let trained = match train_from_texts(instance, &texts, request.trainer.as_ref()) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "train_failed", &msg, None),
    };

    json_response(
        StatusCode::OK,
        &json!({
            "trained": true,
            "added": trained.added_tokens,
            "added_special_tokens": trained.added_special_tokens,
            "vocab_size": effective_vocab_size(instance)
        }),
    )
}

pub async fn handle_train_from_iterator(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: TrainFromIteratorRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let texts = match collect_train_iterator_texts(request.iterator, request.texts) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", &msg, None),
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let trained = match train_from_texts(instance, &texts, request.trainer.as_ref()) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "train_failed", &msg, None),
    };

    json_response(
        StatusCode::OK,
        &json!({
            "trained": true,
            "added": trained.added_tokens,
            "added_special_tokens": trained.added_special_tokens,
            "vocab_size": effective_vocab_size(instance)
        }),
    )
}

pub async fn handle_save(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: SaveRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut instances = state.tokenizer_instances.lock().await;
    let instance = match instances.get_mut(&request.tokenizer_id) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    let mut tokenizer_json = match load_tokenizer_json(instance) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "save_failed",
                &msg,
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    if let Err(msg) = merge_added_tokens_into_json(&mut tokenizer_json, &instance.added_tokens) {
        return json_error(StatusCode::BAD_REQUEST, "save_failed", &msg, None);
    }

    let pretty = request.pretty.unwrap_or(false);
    let serialized = if pretty {
        match serde_json::to_vec_pretty(&tokenizer_json) {
            Ok(v) => v,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "save_failed",
                    &format!("failed to serialize tokenizer JSON: {e}"),
                    None,
                )
            }
        }
    } else {
        match serde_json::to_vec(&tokenizer_json) {
            Ok(v) => v,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "save_failed",
                    &format!("failed to serialize tokenizer JSON: {e}"),
                    None,
                )
            }
        }
    };

    let save_path = match resolve_save_path(&request.path) {
        Ok(v) => v,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "save_failed", &msg, None),
    };

    if save_path.exists() && !request.overwrite.unwrap_or(true) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "save_failed",
            "target path already exists and overwrite=false",
            Some(json!({ "path": save_path.to_string_lossy() })),
        );
    }

    if let Some(parent) = save_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "save_failed",
                &format!("failed to create parent directory: {e}"),
                Some(json!({ "path": save_path.to_string_lossy() })),
            );
        }
    }

    if let Err(e) = std::fs::write(&save_path, &serialized) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "save_failed",
            &format!("failed to write tokenizer JSON: {e}"),
            Some(json!({ "path": save_path.to_string_lossy() })),
        );
    }

    let sha = sha256_hex(&serialized);
    instance.source_kind = "path".to_string();
    instance.source_value = save_path.to_string_lossy().to_string();
    instance.tokenizer_sha256 = sha.clone();

    json_response(
        StatusCode::OK,
        &json!({
            "path": save_path.to_string_lossy(),
            "tokenizer_sha256": sha,
            "bytes": serialized.len()
        }),
    )
}

// ---------------------------------------------------------------------------
// Compare + capabilities
// ---------------------------------------------------------------------------

pub async fn handle_compare(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let request: CompareRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let add_special_tokens = request.add_special_tokens.unwrap_or(true);
    let window = request.window.unwrap_or(8);

    let mut instances = state.tokenizer_instances.lock().await;
    let (left_backend, left_ids, left_hash) = {
        let Some(left) = instances.get_mut(&request.left_tokenizer_id) else {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "left tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.left_tokenizer_id })),
            );
        };
        let left_encoding = match encode_one(left, &request.sequence, add_special_tokens) {
            Ok(v) => v,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "encode_failed", &msg, None),
        };
        (
            left.backend.as_str().to_string(),
            left_encoding.ids.clone(),
            sha256_ids(&left_encoding.ids),
        )
    };

    let (right_backend, right_ids, right_hash) = {
        let Some(right) = instances.get_mut(&request.right_tokenizer_id) else {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "right tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.right_tokenizer_id })),
            );
        };
        let right_encoding = match encode_one(right, &request.sequence, add_special_tokens) {
            Ok(v) => v,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "encode_failed", &msg, None),
        };
        (
            right.backend.as_str().to_string(),
            right_encoding.ids.clone(),
            sha256_ids(&right_encoding.ids),
        )
    };

    let common_prefix = common_prefix_len(&left_ids, &right_ids);
    let first_diff_index = if common_prefix == left_ids.len() && common_prefix == right_ids.len() {
        None
    } else {
        Some(common_prefix)
    };

    let (left_window, right_window) = if let Some(idx) = first_diff_index {
        let start = idx.saturating_sub(window);
        let end_left = usize::min(left_ids.len(), idx + window + 1);
        let end_right = usize::min(right_ids.len(), idx + window + 1);
        (
            left_ids[start..end_left].to_vec(),
            right_ids[start..end_right].to_vec(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    json_response(
        StatusCode::OK,
        &json!({
            "left": {
                "tokenizer_id": request.left_tokenizer_id,
                "backend": left_backend,
                "token_count": left_ids.len(),
                "sha256_ids": left_hash
            },
            "right": {
                "tokenizer_id": request.right_tokenizer_id,
                "backend": right_backend,
                "token_count": right_ids.len(),
                "sha256_ids": right_hash
            },
            "common_prefix": common_prefix,
            "first_diff_index": first_diff_index,
            "left_window": left_window,
            "right_window": right_window
        }),
    )
}

pub async fn handle_capabilities(
    _state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    json_response(
        StatusCode::OK,
        &json!({
            "supported_backends": ["tokenizers", "talu"],
            "supported_options": {
                "tokenizers": {
                    "instances": ["create", "get", "delete"],
                    "encode": ["sequence", "add_special_tokens"],
                    "encode_batch": ["inputs", "add_special_tokens", "padding"],
                    "decode": ["ids", "skip_special_tokens"],
                    "decode_batch": ["ids_batch", "skip_special_tokens"],
                    "vocab": true,
                    "vocab_size": true,
                    "token_to_id": true,
                    "id_to_token": true,
                    "add_tokens": true,
                    "add_special_tokens": true,
                    "train": ["texts", "text", "trainer"],
                    "train_from_iterator": ["iterator", "texts", "trainer"],
                    "save": true
                },
                "talu": {
                    "instances": ["create", "get", "delete"],
                    "encode": ["sequence", "add_special_tokens"],
                    "encode_batch": ["inputs", "add_special_tokens", "padding"],
                    "decode": ["ids", "skip_special_tokens"],
                    "decode_batch": ["ids_batch", "skip_special_tokens"],
                    "vocab": true,
                    "vocab_size": true,
                    "token_to_id": true,
                    "id_to_token": true,
                    "add_tokens": true,
                    "add_special_tokens": true,
                    "train": ["texts", "text", "trainer"],
                    "train_from_iterator": ["iterator", "texts", "trainer"],
                    "save": true
                }
            },
            "unsupported_feature_matrix": {
                "tokenizers": [
                    "is_pretokenized=true",
                    "pair encoding"
                ],
                "talu": [
                    "is_pretokenized=true",
                    "pair encoding"
                ]
            },
            "build": {
                "server": "talu-cli",
                "version": env!("TALU_VERSION")
            }
        }),
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_instance_id(path: &str) -> Option<&str> {
    path.strip_prefix("/v1/tokenizer/instances/")
        .and_then(|tail| tail.split('/').next())
        .filter(|s| !s.is_empty())
}

fn parse_query(query: Option<&str>) -> HashMap<String, String> {
    query
        .map(|q| {
            url::form_urlencoded::parse(q.as_bytes())
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default()
}

async fn parse_json_body<T: for<'de> Deserialize<'de>>(
    req: Request<Incoming>,
) -> Result<T, Response<BoxBody>> {
    let bytes = req
        .into_body()
        .collect()
        .await
        .map_err(|e| {
            json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("failed to read request body: {e}"),
                None,
            )
        })?
        .to_bytes();

    serde_json::from_slice::<T>(&bytes).map_err(|e| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &format!("invalid JSON: {e}"),
            None,
        )
    })
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(
    status: StatusCode,
    code: &str,
    message: &str,
    details: Option<Value>,
) -> Response<BoxBody> {
    json_response(
        status,
        &ErrorEnvelope {
            error: ErrorBody {
                code: code.to_string(),
                message: message.to_string(),
                details,
            },
        },
    )
}

fn normalize_input_sequence(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut tokens = Vec::with_capacity(parts.len());
            for part in parts {
                let Some(token) = part.as_str() else {
                    return Err("sequence array must contain only strings".to_string());
                };
                tokens.push(token);
            }
            Ok(tokens.join(" "))
        }
        _ => Err("sequence must be a string or array of strings".to_string()),
    }
}

fn normalize_batch_inputs(inputs: &[Value]) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(inputs.len());
    for item in inputs {
        out.push(normalize_input_sequence(item)?);
    }
    Ok(out)
}

fn parse_added_token_request(
    token: AddedTokenRequest,
    default_special_normalized: bool,
) -> Result<ParsedAddedToken, String> {
    match token {
        AddedTokenRequest::Text(content) => {
            if content.is_empty() {
                return Err("token content must be non-empty".to_string());
            }
            Ok(ParsedAddedToken {
                content,
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: if default_special_normalized {
                    false
                } else {
                    true
                },
            })
        }
        AddedTokenRequest::Object(obj) => {
            if obj.content.is_empty() {
                return Err("token content must be non-empty".to_string());
            }
            Ok(ParsedAddedToken {
                content: obj.content,
                single_word: obj.single_word.unwrap_or(false),
                lstrip: obj.lstrip.unwrap_or(false),
                rstrip: obj.rstrip.unwrap_or(false),
                normalized: obj.normalized.unwrap_or(!default_special_normalized),
            })
        }
    }
}

fn extract_special_tokens_from_value(value: &Value) -> Result<Vec<ParsedAddedToken>, String> {
    let mut out = Vec::new();

    match value {
        Value::String(s) => {
            push_special_token_text(&mut out, s)?;
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(s) => push_special_token_text(&mut out, s)?,
                    Value::Object(_) => {
                        let token: AddedTokenRequest = serde_json::from_value(item.clone())
                            .map_err(|e| format!("invalid special token entry: {e}"))?;
                        out.push(parse_added_token_request(token, true)?);
                    }
                    _ => {
                        return Err("special_tokens array must contain strings or token objects"
                            .to_string())
                    }
                }
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                match v {
                    Value::String(s) => push_special_token_text(&mut out, s)?,
                    Value::Array(arr) => {
                        for item in arr {
                            match item {
                                Value::String(s) => push_special_token_text(&mut out, s)?,
                                Value::Object(_) => {
                                    let token: AddedTokenRequest = serde_json::from_value(item.clone())
                                        .map_err(|e| format!("invalid special token entry: {e}"))?;
                                    out.push(parse_added_token_request(token, true)?);
                                }
                                _ => {
                                    return Err(
                                        "special_tokens object values must be strings or arrays of strings/token objects"
                                            .to_string(),
                                    )
                                }
                            }
                        }
                    }
                    Value::Object(_) => {
                        let token: AddedTokenRequest = serde_json::from_value(v.clone())
                            .map_err(|e| format!("invalid special token entry: {e}"))?;
                        out.push(parse_added_token_request(token, true)?);
                    }
                    _ => {
                        return Err(
                            "special_tokens object values must be strings or arrays of strings/token objects"
                                .to_string(),
                        )
                    }
                }
            }
        }
        _ => return Err("special_tokens must be a string, array, or object".to_string()),
    }

    Ok(out)
}

fn push_special_token_text(out: &mut Vec<ParsedAddedToken>, content: &str) -> Result<(), String> {
    if content.is_empty() {
        return Err("special token content must be non-empty".to_string());
    }
    out.push(ParsedAddedToken {
        content: content.to_string(),
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: false,
    });
    Ok(())
}

fn token_exists(instance: &mut TokenizerInstance, content: &str) -> bool {
    if instance
        .added_tokens
        .iter()
        .any(|added| added.content == content)
    {
        return true;
    }

    // SAFETY: handle is valid; content pointer valid during call.
    let base_or_dynamic = unsafe {
        talu_sys::talu_tokenizer_token_to_id(
            instance.handle.as_ptr(),
            content.as_bytes().as_ptr(),
            content.len(),
        )
    };
    base_or_dynamic >= 0
}

fn next_added_token_id(instance: &TokenizerInstance) -> u32 {
    let mut next = u32::try_from(instance.vocab_size).unwrap_or(u32::MAX);
    for added in &instance.added_tokens {
        next = next.max(added.id.saturating_add(1));
    }
    next
}

fn effective_vocab_size(instance: &TokenizerInstance) -> usize {
    instance
        .vocab_size
        .saturating_add(instance.added_tokens.len())
}

fn load_tokenizer_json(instance: &TokenizerInstance) -> Result<Value, String> {
    if instance.source_kind == "json" {
        return serde_json::from_str(&instance.source_value)
            .map_err(|e| format!("failed to parse tokenizer JSON source: {e}"));
    }

    if instance.source_kind != "path" {
        return Err(format!("unsupported source.kind: {}", instance.source_kind));
    }

    let source_path = Path::new(&instance.source_value);
    let candidates = if source_path.is_dir() {
        vec![source_path.join("tokenizer.json")]
    } else {
        vec![
            source_path.to_path_buf(),
            source_path.join("tokenizer.json"),
        ]
    };

    for candidate in candidates {
        if let Ok(bytes) = std::fs::read(&candidate) {
            let parsed = serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| format!("failed to parse {}: {e}", candidate.display()))?;
            return Ok(parsed);
        }
    }

    Err("unable to load tokenizer JSON from current source path".to_string())
}

fn resolve_save_path(path: &str) -> Result<std::path::PathBuf, String> {
    if path.is_empty() {
        return Err("path must be non-empty".to_string());
    }

    let requested = std::path::PathBuf::from(path);
    if requested.exists() && requested.is_dir() {
        return Ok(requested.join("tokenizer.json"));
    }
    if path.ends_with(std::path::MAIN_SEPARATOR) {
        return Ok(requested.join("tokenizer.json"));
    }
    if requested.extension().is_none() {
        return Ok(requested.join("tokenizer.json"));
    }
    Ok(requested)
}

fn merge_added_tokens_into_json(
    tokenizer_json: &mut Value,
    added_tokens: &[AddedTokenEntry],
) -> Result<(), String> {
    let root = tokenizer_json
        .as_object_mut()
        .ok_or_else(|| "tokenizer JSON root must be an object".to_string())?;

    let mut merged: Vec<Value> = Vec::new();
    let mut seen = std::collections::HashSet::<String>::new();

    if let Some(existing) = root.get("added_tokens") {
        let arr = existing
            .as_array()
            .ok_or_else(|| "tokenizer JSON field added_tokens must be an array".to_string())?;
        for item in arr {
            if let Some(content) = item.get("content").and_then(Value::as_str) {
                seen.insert(content.to_string());
            }
            merged.push(item.clone());
        }
    }

    for added in added_tokens {
        if seen.contains(&added.content) {
            continue;
        }
        seen.insert(added.content.clone());
        merged.push(json!({
            "id": added.id,
            "content": added.content,
            "single_word": added.single_word,
            "lstrip": added.lstrip,
            "rstrip": added.rstrip,
            "normalized": added.normalized,
            "special": added.special
        }));
    }

    merged.sort_by_key(|item| item.get("id").and_then(Value::as_u64).unwrap_or(u64::MAX));
    root.insert("added_tokens".to_string(), Value::Array(merged));
    Ok(())
}

#[derive(Debug, Default)]
struct TrainResult {
    added_tokens: usize,
    added_special_tokens: usize,
}

fn collect_train_texts(
    texts: Option<Vec<String>>,
    text: Option<String>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    if let Some(mut many) = texts {
        out.append(&mut many);
    }
    if let Some(one) = text {
        out.push(one);
    }
    if out.is_empty() {
        return Err("train requires texts or text".to_string());
    }
    Ok(out)
}

fn collect_train_iterator_texts(
    iterator: Option<Vec<String>>,
    texts: Option<Vec<String>>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    if let Some(mut values) = iterator {
        out.append(&mut values);
    }
    if let Some(mut values) = texts {
        out.append(&mut values);
    }
    if out.is_empty() {
        return Err("train_from_iterator requires iterator or texts".to_string());
    }
    Ok(out)
}

fn train_from_texts(
    instance: &mut TokenizerInstance,
    texts: &[String],
    trainer: Option<&TrainerRequest>,
) -> Result<TrainResult, String> {
    if texts.is_empty() {
        return Err("training input is empty".to_string());
    }

    let mut result = TrainResult::default();
    let mut remaining_capacity = trainer.and_then(|t| {
        t.vocab_size
            .map(|target| target.saturating_sub(effective_vocab_size(instance)))
    });

    if let Some(special_tokens_value) = trainer.and_then(|t| t.special_tokens.as_ref()) {
        let special_tokens = extract_special_tokens_from_value(special_tokens_value)?;
        for token in special_tokens {
            if token_exists(instance, &token.content) {
                continue;
            }
            if remaining_capacity == Some(0) {
                break;
            }
            let id = next_added_token_id(instance);
            instance.added_tokens.push(AddedTokenEntry {
                content: token.content,
                id,
                special: true,
                single_word: token.single_word,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
                normalized: token.normalized,
            });
            result.added_special_tokens += 1;
            if let Some(rem) = remaining_capacity.as_mut() {
                *rem = rem.saturating_sub(1);
            }
        }
    }

    let mut freq: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for token in text.split_whitespace() {
            if token.is_empty() {
                continue;
            }
            *freq.entry(token.to_string()).or_insert(0) += 1;
        }
    }

    let mut candidates: Vec<(String, usize)> = freq.into_iter().collect();
    candidates.sort_by(|(tok_a, freq_a), (tok_b, freq_b)| {
        freq_b.cmp(freq_a).then_with(|| tok_a.cmp(tok_b))
    });

    for (content, _) in candidates {
        if token_exists(instance, &content) {
            continue;
        }
        if remaining_capacity == Some(0) {
            break;
        }
        let id = next_added_token_id(instance);
        instance.added_tokens.push(AddedTokenEntry {
            content,
            id,
            special: false,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
        });
        result.added_tokens += 1;
        if let Some(rem) = remaining_capacity.as_mut() {
            *rem = rem.saturating_sub(1);
        }
    }

    Ok(result)
}

fn build_encode_options(
    add_special_tokens: bool,
    truncation: Option<&InstanceTruncationConfig>,
) -> talu_sys::EncodeOptions {
    let mut options = talu_sys::EncodeOptions::default();
    options.add_bos = if add_special_tokens { 1 } else { 0 };
    options.add_eos = if add_special_tokens { 1 } else { 0 };
    if let Some(config) = truncation {
        options.truncation = 1;
        options.max_length = config.max_length;
        options.truncation_side = match config.direction {
            TruncationDirection::Right => 0,
            TruncationDirection::Left => 1,
        };
    }
    options
}

fn encode_one(
    instance: &mut TokenizerInstance,
    text: &str,
    add_special_tokens: bool,
) -> Result<MutableEncoding, String> {
    let options = build_encode_options(add_special_tokens, instance.truncation.as_ref());

    // SAFETY: tokenizer handle is valid while instance exists; text pointer valid for call.
    let result = unsafe {
        talu_sys::talu_tokenizer_encode(
            instance.handle.as_ptr(),
            text.as_bytes().as_ptr(),
            text.len(),
            &options,
        )
    };

    if !result.error_msg.is_null() {
        return Err(ffi_message(result.error_msg, "encode failed"));
    }

    let ids = copy_u32_slice(result.ids, result.num_tokens);
    let attention_mask = copy_u32_slice(result.attention_mask, result.num_tokens);
    let special_tokens_mask = copy_u32_slice(result.special_tokens_mask, result.num_tokens);
    let offsets = copy_offsets_slice(result.offsets, result.num_tokens);

    // SAFETY: result was returned by talu_tokenizer_encode and must be freed once.
    unsafe { talu_sys::talu_encode_result_free(result) };

    let tokens = ids_to_tokens(instance, &ids)?;
    let type_ids = vec![0u32; ids.len()];

    Ok(MutableEncoding {
        ids,
        tokens,
        type_ids,
        attention_mask,
        special_tokens_mask,
        offsets,
    })
}

fn decode_ids(
    instance: &mut TokenizerInstance,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String, String> {
    let options = talu_sys::DecodeOptionsC {
        skip_special_tokens: if skip_special_tokens { 1 } else { 0 },
    };

    // SAFETY: handle valid; ids pointer is valid for ids.len elements.
    let result = unsafe {
        talu_sys::talu_tokenizer_decode(instance.handle.as_ptr(), ids.as_ptr(), ids.len(), &options)
    };

    if !result.error_msg.is_null() {
        return Err(ffi_message(result.error_msg, "decode failed"));
    }

    // SAFETY: decode result points to UTF-8 bytes returned by C API.
    let text_bytes = unsafe { std::slice::from_raw_parts(result.text, result.text_len) };
    let text = String::from_utf8_lossy(text_bytes).to_string();

    // SAFETY: free decode result buffers from C API.
    unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };

    Ok(text)
}

fn decode_ids_with_added_fallback(
    instance: &mut TokenizerInstance,
    ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String, String> {
    match decode_ids(instance, ids, skip_special_tokens) {
        Ok(text) => Ok(text),
        Err(primary_err) => {
            let mut out = String::new();
            for &id in ids {
                let id_i32 = i32::try_from(id).map_err(|_| primary_err.clone())?;
                if let Some(added) = instance.added_tokens.iter().find(|t| t.id == id) {
                    if skip_special_tokens && added.special {
                        continue;
                    }
                    out.push_str(&added.content);
                    continue;
                }

                let token = id_to_token(instance, id_i32).map_err(|_| primary_err.clone())?;
                out.push_str(&token);
            }
            Ok(out)
        }
    }
}

fn ids_to_tokens(instance: &mut TokenizerInstance, ids: &[u32]) -> Result<Vec<String>, String> {
    let mut tokens = Vec::with_capacity(ids.len());
    for &id in ids {
        let token = id_to_token(instance, i32::try_from(id).unwrap_or(-1))?;
        tokens.push(token);
    }
    Ok(tokens)
}

fn id_to_token(instance: &mut TokenizerInstance, token_id: i32) -> Result<String, String> {
    let mut out_ptr: *mut c_char = std::ptr::null_mut();
    // SAFETY: out pointer is valid for write and handle is valid.
    let rc = unsafe {
        talu_sys::talu_tokenizer_id_to_token(
            instance.handle.as_ptr(),
            token_id,
            &mut out_ptr as *mut _ as *mut c_void,
        )
    };

    if rc != 0 {
        return Err(take_last_error()
            .unwrap_or_else(|| format!("token id {token_id} is not in vocabulary")));
    }

    if out_ptr.is_null() {
        return Err(format!("token id {token_id} returned null token"));
    }

    // SAFETY: pointer is owned C string from core.
    let token = unsafe { CStr::from_ptr(out_ptr) }
        .to_string_lossy()
        .to_string();

    // SAFETY: free string allocated by core tokenizer API.
    unsafe { talu_sys::talu_text_free(out_ptr) };

    Ok(token)
}

fn id_to_token_with_added(
    instance: &mut TokenizerInstance,
    token_id: i32,
) -> Result<String, String> {
    if token_id >= 0 {
        if let Some(added) = instance
            .added_tokens
            .iter()
            .find(|added| added.id == token_id as u32)
        {
            return Ok(added.content.clone());
        }
    }
    id_to_token(instance, token_id)
}

fn special_tokens(instance: &mut TokenizerInstance) -> talu_sys::SpecialTokensResult {
    // SAFETY: handle valid while instance is locked.
    unsafe { talu_sys::talu_tokenizer_get_special_tokens(instance.handle.as_ptr()) }
}

fn build_effective_padding(
    instance: &mut TokenizerInstance,
    req: Option<PaddingRequest>,
) -> Result<Option<InstancePaddingConfig>, Response<BoxBody>> {
    let Some(req_padding) = req else {
        return Ok(instance.padding.clone());
    };

    if req_padding.enabled == Some(false) {
        return Ok(None);
    }

    if req_padding.multiple_of == Some(0) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "padding.multiple_of must be greater than zero",
            Some(json!({ "field": "padding.multiple_of" })),
        ));
    }

    let direction_str = req_padding
        .direction
        .as_deref()
        .or_else(|| instance.padding.as_ref().map(|p| p.direction.as_str()))
        .unwrap_or("right");

    let Some(direction) = PaddingDirection::from_str(direction_str) else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "padding.direction must be one of: left, right",
            Some(json!({ "field": "padding.direction" })),
        ));
    };

    let special = special_tokens(instance);
    let inherited_pad = instance.padding.clone();

    let pad_id = req_padding
        .pad_id
        .or_else(|| inherited_pad.as_ref().map(|p| p.pad_id))
        .or_else(|| u32::try_from(special.pad_token_id).ok())
        .unwrap_or(0);

    let pad_type_id = req_padding
        .pad_type_id
        .or_else(|| inherited_pad.as_ref().map(|p| p.pad_type_id))
        .unwrap_or(0);

    let pad_token = if let Some(v) = req_padding.pad_token {
        v
    } else if let Some(v) = inherited_pad.as_ref().map(|p| p.pad_token.clone()) {
        v
    } else {
        id_to_token(instance, i32::try_from(pad_id).unwrap_or(-1))
            .unwrap_or_else(|_| "<PAD>".to_string())
    };

    Ok(Some(InstancePaddingConfig {
        direction,
        pad_id,
        pad_type_id,
        pad_token,
        length: req_padding
            .length
            .or_else(|| inherited_pad.as_ref().and_then(|p| p.length)),
        multiple_of: req_padding
            .multiple_of
            .or_else(|| inherited_pad.as_ref().and_then(|p| p.multiple_of)),
    }))
}

fn apply_padding_to_batch(rows: &mut [MutableEncoding], config: &InstancePaddingConfig) {
    let max_len = rows.iter().map(|row| row.ids.len()).max().unwrap_or(0);
    let mut target_len = config.length.unwrap_or(max_len);
    if target_len < max_len {
        target_len = max_len;
    }
    if let Some(multiple_of) = config.multiple_of {
        if multiple_of > 0 {
            let rem = target_len % multiple_of;
            if rem != 0 {
                target_len += multiple_of - rem;
            }
        }
    }

    for row in rows {
        apply_padding_to_row(row, config, target_len);
    }
}

fn apply_padding_to_encoding(row: &mut MutableEncoding, config: &InstancePaddingConfig) {
    let mut target_len = config.length.unwrap_or(row.ids.len());
    if target_len < row.ids.len() {
        target_len = row.ids.len();
    }
    if let Some(multiple_of) = config.multiple_of {
        if multiple_of > 0 {
            let rem = target_len % multiple_of;
            if rem != 0 {
                target_len += multiple_of - rem;
            }
        }
    }
    apply_padding_to_row(row, config, target_len);
}

fn apply_padding_to_row(
    row: &mut MutableEncoding,
    config: &InstancePaddingConfig,
    target_len: usize,
) {
    if row.ids.len() >= target_len {
        return;
    }

    let pad_count = target_len - row.ids.len();
    match config.direction {
        PaddingDirection::Right => {
            row.ids
                .extend(std::iter::repeat_n(config.pad_id, pad_count));
            row.tokens
                .extend(std::iter::repeat_n(config.pad_token.clone(), pad_count));
            row.type_ids
                .extend(std::iter::repeat_n(config.pad_type_id, pad_count));
            row.attention_mask.extend(std::iter::repeat_n(0, pad_count));
            row.special_tokens_mask
                .extend(std::iter::repeat_n(1, pad_count));
            row.offsets.extend(std::iter::repeat_n([0, 0], pad_count));
        }
        PaddingDirection::Left => {
            prepend_repeat(&mut row.ids, config.pad_id, pad_count);
            prepend_repeat(&mut row.tokens, config.pad_token.clone(), pad_count);
            prepend_repeat(&mut row.type_ids, config.pad_type_id, pad_count);
            prepend_repeat(&mut row.attention_mask, 0, pad_count);
            prepend_repeat(&mut row.special_tokens_mask, 1, pad_count);
            prepend_repeat(&mut row.offsets, [0, 0], pad_count);
        }
    }
}

fn prepend_repeat<T: Clone>(vec: &mut Vec<T>, value: T, count: usize) {
    if count == 0 {
        return;
    }
    let mut prefix = vec![value; count];
    prefix.extend(vec.iter().cloned());
    *vec = prefix;
}

fn encoding_payload_from_mutable(
    row: &MutableEncoding,
    return_fields: &ReturnFields,
) -> EncodingPayload {
    EncodingPayload {
        ids: return_fields.include_ids().then(|| row.ids.clone()),
        tokens: return_fields.include_tokens().then(|| row.tokens.clone()),
        type_ids: return_fields
            .include_type_ids()
            .then(|| row.type_ids.clone()),
        attention_mask: return_fields
            .include_attention_mask()
            .then(|| row.attention_mask.clone()),
        special_tokens_mask: return_fields
            .include_special_tokens_mask()
            .then(|| row.special_tokens_mask.clone()),
        offsets: return_fields.include_offsets().then(|| row.offsets.clone()),
    }
}

fn copy_u32_slice(ptr: *mut u32, len: usize) -> Vec<u32> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller guarantees ptr points to len elements.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn copy_offsets_slice(ptr: *mut talu_sys::TokenOffset, len: usize) -> Vec<[u32; 2]> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller guarantees ptr points to len elements.
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .iter()
        .map(|off| [off.start, off.end])
        .collect()
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    let mut idx = 0usize;
    let max = usize::min(a.len(), b.len());
    while idx < max && a[idx] == b[idx] {
        idx += 1;
    }
    idx
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn tokenizer_sha256_from_path_source(source_value: &str, handle_ptr: *mut c_void) -> String {
    let source_path = Path::new(source_value);

    if let Ok(bytes) = std::fs::read(source_path) {
        return sha256_hex(&bytes);
    }
    if let Ok(bytes) = std::fs::read(source_path.join("tokenizer.json")) {
        return sha256_hex(&bytes);
    }
    if let Some(model_dir) = tokenizer_model_dir(handle_ptr) {
        let model_path = Path::new(&model_dir);
        if let Ok(bytes) = std::fs::read(model_path.join("tokenizer.json")) {
            return sha256_hex(&bytes);
        }
    }

    sha256_hex(source_value.as_bytes())
}

fn tokenizer_model_dir(handle_ptr: *mut c_void) -> Option<String> {
    let mut out_ptr: *mut c_char = std::ptr::null_mut();
    // SAFETY: handle pointer is a valid tokenizer handle and out pointer is valid for write.
    let rc = unsafe {
        talu_sys::talu_tokenizer_get_model_dir(handle_ptr, &mut out_ptr as *mut _ as *mut c_void)
    };
    if rc != 0 || out_ptr.is_null() {
        return None;
    }

    // SAFETY: C API returns a NUL-terminated allocated string.
    let model_dir = unsafe { CStr::from_ptr(out_ptr) }
        .to_string_lossy()
        .to_string();
    // SAFETY: free string allocated by C API.
    unsafe { talu_sys::talu_text_free(out_ptr) };

    if model_dir.is_empty() {
        None
    } else {
        Some(model_dir)
    }
}

fn sha256_ids(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    hex_lower(&hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn ffi_message(error_msg_ptr: *const u8, fallback: &str) -> String {
    if !error_msg_ptr.is_null() {
        // SAFETY: C API returns NUL-terminated error strings.
        return unsafe { CStr::from_ptr(error_msg_ptr as *const c_char) }
            .to_string_lossy()
            .to_string();
    }
    take_last_error().unwrap_or_else(|| fallback.to_string())
}

fn take_last_error() -> Option<String> {
    // SAFETY: returns thread-local pointer or null.
    let ptr = unsafe { talu_sys::talu_last_error() };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: non-null C string pointer from C API.
    let msg = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string();
    // SAFETY: clear thread-local error state.
    unsafe { talu_sys::talu_clear_error() };
    Some(msg)
}
