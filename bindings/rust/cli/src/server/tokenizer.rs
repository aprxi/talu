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
type TokenizerInstanceHandle = Arc<tokio::sync::Mutex<TokenizerInstance>>;

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
    #[serde(default)]
    include_hash: Option<bool>,
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
    #[serde(default)]
    include_hash: Option<bool>,
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
    fn ids_only() -> Self {
        Self {
            ids: Some(true),
            tokens: Some(false),
            type_ids: Some(false),
            attention_mask: Some(false),
            special_tokens_mask: Some(false),
            offsets: Some(false),
        }
    }

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
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256_ids: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256_ids_batch: Option<Vec<String>>,
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

#[derive(Debug, Clone, Copy)]
struct EncodeBuildFlags {
    include_tokens: bool,
    include_type_ids: bool,
    include_attention_mask: bool,
    include_special_tokens_mask: bool,
    include_offsets: bool,
}

impl EncodeBuildFlags {
    fn from_return_fields(return_fields: &ReturnFields) -> Self {
        Self {
            include_tokens: return_fields.include_tokens(),
            include_type_ids: return_fields.include_type_ids(),
            include_attention_mask: return_fields.include_attention_mask(),
            include_special_tokens_mask: return_fields.include_special_tokens_mask(),
            include_offsets: return_fields.include_offsets(),
        }
    }
}

fn ids_only_build_flags() -> EncodeBuildFlags {
    EncodeBuildFlags {
        include_tokens: false,
        include_type_ids: false,
        include_attention_mask: false,
        include_special_tokens_mask: false,
        include_offsets: false,
    }
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
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_backend",
                "backend must be one of: tokenizers, talu",
                Some(json!({
                    "field": "backend",
                    "reason": "unsupported_backend",
                    "expected": "one of: tokenizers, talu",
                    "endpoint": "/v1/tokenizer/instances"
                })),
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

    if source_kind == "path" {
        if let Err(resp) = enforce_path_source_policy(&source_value) {
            return resp;
        }
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
    instances.insert(
        tokenizer_id.clone(),
        Arc::new(tokio::sync::Mutex::new(instance)),
    );

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

    let instance_handle = match find_tokenizer_instance(&state, tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": tokenizer_id })),
            );
        }
    };
    let instance = instance_handle.lock().await;

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
    let wants_binary = request_wants_binary_response(req.headers());
    let request: EncodeRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if request.pair.as_ref().is_some_and(|v| !v.is_null()) {
        return json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_option",
            "pair encoding is not supported by /v1/tokenizer/encode in this build",
            Some(json!({
                "field": "pair",
                "reason": "pair_encoding_unsupported",
                "expected": "pair must be null or omitted",
                "endpoint": "/v1/tokenizer/encode"
            })),
        );
    }

    let is_pretokenized = request.is_pretokenized.unwrap_or(false);
    let sequence = match normalize_input_sequence_with_mode(&request.sequence, is_pretokenized) {
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
    let return_fields = request.return_fields.unwrap_or_else(|| {
        if wants_binary {
            ReturnFields::ids_only()
        } else {
            ReturnFields::default()
        }
    });
    let include_timing = request.benchmark.unwrap_or(false);
    if wants_binary && request.include_hash == Some(true) {
        return json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_option",
            "include_hash=true is not supported for binary tokenizer responses",
            Some(json!({
                "field": "include_hash",
                "endpoint": "/v1/tokenizer/encode"
            })),
        );
    }
    if wants_binary {
        if let Err(resp) =
            validate_binary_ids_only_return_fields(&return_fields, "/v1/tokenizer/encode")
        {
            return resp;
        }
    }
    let include_hash = request.include_hash.unwrap_or(!wants_binary);
    let build_flags = if wants_binary {
        ids_only_build_flags()
    } else {
        EncodeBuildFlags::from_return_fields(&return_fields)
    };

    let total_start = Instant::now();

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let encode_start = Instant::now();
    let mut encoding = match encode_one(&mut instance, &sequence, add_special_tokens, build_flags) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "encode_failed",
                &msg,
                Some(json!({ "endpoint": "/v1/tokenizer/encode" })),
            )
        }
    };

    if let Some(config) = instance.padding.clone() {
        apply_padding_to_encoding(&mut encoding, &config);
    }

    let encode_ms = elapsed_ms(encode_start);
    let total_ms = elapsed_ms(total_start);
    if wants_binary {
        let ids_count = encoding.ids.len();
        let body = encode_ids_binary_v1(&encoding.ids);

        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/octet-stream")
            .header("x-talu-binary-format", "ids_le_u32_v1")
            .header("x-talu-ids-count", ids_count.to_string())
            .header(
                "x-talu-impl",
                format!("{}.encode", instance.backend.as_str()),
            );

        if include_timing {
            response = response
                .header("x-talu-timing-encode-ms", format!("{encode_ms:.3}"))
                .header("x-talu-timing-total-ms", format!("{total_ms:.3}"));
        }

        return response
            .body(Full::new(Bytes::from(body)).boxed())
            .unwrap_or_else(|_| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "failed to build binary tokenizer response",
                    Some(json!({ "endpoint": "/v1/tokenizer/encode" })),
                )
            });
    }

    let sha256_ids = include_hash.then(|| sha256_ids(&encoding.ids));
    let payload = encoding_payload_from_owned(encoding, &return_fields);

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
    let wants_binary = request_wants_binary_response(req.headers());
    let request: EncodeBatchRequest = match parse_json_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let is_pretokenized = request.is_pretokenized.unwrap_or(false);
    let inputs = match normalize_batch_inputs_with_mode(&request.inputs, is_pretokenized) {
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
    let return_fields = request.return_fields.unwrap_or_else(|| {
        if wants_binary {
            ReturnFields::ids_only()
        } else {
            ReturnFields::default()
        }
    });
    let include_timing = request.benchmark.unwrap_or(false);
    if wants_binary && request.include_hash == Some(true) {
        return json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_option",
            "include_hash=true is not supported for binary tokenizer responses",
            Some(json!({
                "field": "include_hash",
                "endpoint": "/v1/tokenizer/encode_batch"
            })),
        );
    }
    if wants_binary {
        if let Err(resp) =
            validate_binary_ids_only_return_fields(&return_fields, "/v1/tokenizer/encode_batch")
        {
            return resp;
        }
    }
    let include_hash = request.include_hash.unwrap_or(!wants_binary);
    let build_flags = if wants_binary {
        ids_only_build_flags()
    } else {
        EncodeBuildFlags::from_return_fields(&return_fields)
    };

    let total_start = Instant::now();

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let effective_padding = match build_effective_padding(&mut instance, request.padding) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let encode_start = Instant::now();
    let mut rows = Vec::with_capacity(inputs.len());
    for text in &inputs {
        let encoding = match encode_one(&mut instance, text, add_special_tokens, build_flags) {
            Ok(v) => v,
            Err(msg) => {
                return json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "encode_failed",
                    &msg,
                    Some(json!({ "endpoint": "/v1/tokenizer/encode_batch" })),
                )
            }
        };
        rows.push(encoding);
    }

    let mut mutable_rows: Vec<MutableEncoding> = rows;

    if let Some(config) = effective_padding.as_ref() {
        apply_padding_to_batch(&mut mutable_rows, config);
    }

    let encode_ms = elapsed_ms(encode_start);
    let total_ms = elapsed_ms(total_start);
    if wants_binary {
        let num_sequences = mutable_rows.len();
        let total_ids = mutable_rows.iter().map(|row| row.ids.len()).sum::<usize>();
        let body = encode_batch_ids_binary_v1(&mutable_rows);

        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/octet-stream")
            .header("x-talu-binary-format", "ids_batch_le_u32_v1")
            .header("x-talu-num-sequences", num_sequences.to_string())
            .header("x-talu-total-ids", total_ids.to_string())
            .header(
                "x-talu-impl",
                format!("{}.encode_batch", instance.backend.as_str()),
            );

        if include_timing {
            response = response
                .header("x-talu-timing-encode-ms", format!("{encode_ms:.3}"))
                .header("x-talu-timing-total-ms", format!("{total_ms:.3}"));
        }

        return response
            .body(Full::new(Bytes::from(body)).boxed())
            .unwrap_or_else(|_| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "failed to build binary tokenizer batch response",
                    Some(json!({ "endpoint": "/v1/tokenizer/encode_batch" })),
                )
            });
    }

    let lengths: Vec<usize> = mutable_rows.iter().map(|row| row.ids.len()).collect();

    let num_sequences = mutable_rows.len();

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
    let sha256_ids_batch = include_hash.then(|| {
        mutable_rows
            .iter()
            .map(|row| sha256_ids(&row.ids))
            .collect::<Vec<_>>()
    });
    let encodings: Vec<EncodingPayload> = mutable_rows
        .into_iter()
        .map(|row| encoding_payload_from_owned(row, &return_fields))
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
                num_sequences,
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let text = match decode_ids(&mut instance, &request.ids, skip_special_tokens) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "decode_failed",
                &msg,
                Some(json!({ "endpoint": "/v1/tokenizer/decode" })),
            )
        }
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let mut texts = Vec::with_capacity(request.ids_batch.len());
    for ids in &request.ids_batch {
        let text = match decode_ids(&mut instance, ids, skip_special_tokens) {
            Ok(v) => v,
            Err(msg) => {
                return json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "decode_failed",
                    &msg,
                    Some(json!({ "endpoint": "/v1/tokenizer/decode_batch" })),
                )
            }
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

    let instance_handle = match find_tokenizer_instance(&state, tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": tokenizer_id })),
            )
        }
    };
    let instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": tokenizer_id })),
            )
        }
    };
    let instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let token = match id_to_token_with_added(&mut instance, request.token_id) {
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
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_option",
            "stride is not supported by this tokenizer backend",
            Some(json!({
                "field": "stride",
                "reason": "unsupported_option",
                "expected": "stride must be 0 or omitted",
                "endpoint": "/v1/tokenizer/enable_truncation"
            })),
        );
    }

    if let Some(strategy) = request.strategy.as_deref() {
        if strategy != "longest_first" {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_option",
                "only strategy=longest_first is supported",
                Some(json!({
                    "field": "strategy",
                    "reason": "unsupported_option",
                    "expected": "strategy=longest_first",
                    "got": strategy,
                    "endpoint": "/v1/tokenizer/enable_truncation"
                })),
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let special = special_tokens(&mut instance);
    let pad_id = request
        .pad_id
        .or_else(|| u32::try_from(special.pad_token_id).ok())
        .unwrap_or(0);
    let pad_token = match request.pad_token {
        Some(v) => v,
        None => id_to_token(&mut instance, i32::try_from(pad_id).unwrap_or(-1))
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

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

        if token_exists(&mut instance, &parsed.content) {
            continue;
        }

        let id = next_added_token_id(&instance);
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
            "vocab_size": effective_vocab_size(&instance)
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let mut added = 0usize;
    for parsed in parsed_tokens {
        if token_exists(&mut instance, &parsed.content) {
            continue;
        }
        let id = next_added_token_id(&instance);
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
            "vocab_size": effective_vocab_size(&instance)
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let trained = match train_from_texts(&mut instance, &texts, request.trainer.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "train_failed",
                &msg,
                Some(json!({ "endpoint": "/v1/tokenizer/train" })),
            )
        }
    };

    json_response(
        StatusCode::OK,
        &json!({
            "trained": true,
            "added": trained.added_tokens,
            "added_special_tokens": trained.added_special_tokens,
            "vocab_size": effective_vocab_size(&instance)
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let trained = match train_from_texts(&mut instance, &texts, request.trainer.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "train_failed",
                &msg,
                Some(json!({ "endpoint": "/v1/tokenizer/train_from_iterator" })),
            )
        }
    };

    json_response(
        StatusCode::OK,
        &json!({
            "trained": true,
            "added": trained.added_tokens,
            "added_special_tokens": trained.added_special_tokens,
            "vocab_size": effective_vocab_size(&instance)
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

    let instance_handle = match find_tokenizer_instance(&state, &request.tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };
    let mut instance = instance_handle.lock().await;

    let mut tokenizer_json = match load_tokenizer_json(&instance) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "save_failed",
                &msg,
                Some(json!({ "tokenizer_id": request.tokenizer_id })),
            )
        }
    };

    if let Err(msg) = merge_added_tokens_into_json(&mut tokenizer_json, &instance.added_tokens) {
        return json_error(StatusCode::UNPROCESSABLE_ENTITY, "save_failed", &msg, None);
    }

    let pretty = request.pretty.unwrap_or(false);
    let serialized = if pretty {
        match serde_json::to_vec_pretty(&tokenizer_json) {
            Ok(v) => v,
            Err(e) => {
                return json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
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
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "save_failed",
                    &format!("failed to serialize tokenizer JSON: {e}"),
                    None,
                )
            }
        }
    };

    let save_path = match resolve_save_path(&request.path) {
        Ok(v) => v,
        Err(msg) => {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "save_failed",
                &msg,
                Some(json!({
                    "field": "path",
                    "endpoint": "/v1/tokenizer/save"
                })),
            )
        }
    };

    if save_path.exists() && !request.overwrite.unwrap_or(true) {
        return json_error(
            StatusCode::CONFLICT,
            "conflict",
            "target path already exists and overwrite=false",
            Some(json!({
                "field": "path",
                "reason": "path_exists_overwrite_false",
                "got": save_path.to_string_lossy(),
                "endpoint": "/v1/tokenizer/save"
            })),
        );
    }

    if let Some(parent) = save_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "save_failed",
                &format!("failed to create parent directory: {e}"),
                Some(json!({
                    "field": "path",
                    "reason": "create_parent_failed",
                    "got": save_path.to_string_lossy(),
                    "endpoint": "/v1/tokenizer/save"
                })),
            );
        }
    }

    if let Err(e) = std::fs::write(&save_path, &serialized) {
        return json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "save_failed",
            &format!("failed to write tokenizer JSON: {e}"),
            Some(json!({
                "field": "path",
                "reason": "write_failed",
                "got": save_path.to_string_lossy(),
                "endpoint": "/v1/tokenizer/save"
            })),
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
    let compare_build_flags = EncodeBuildFlags {
        include_tokens: false,
        include_type_ids: false,
        include_attention_mask: false,
        include_special_tokens_mask: false,
        include_offsets: false,
    };

    let left_handle = match find_tokenizer_instance(&state, &request.left_tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "left tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.left_tokenizer_id })),
            )
        }
    };
    let right_handle = match find_tokenizer_instance(&state, &request.right_tokenizer_id).await {
        Some(handle) => handle,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "tokenizer_not_found",
                "right tokenizer instance not found",
                Some(json!({ "tokenizer_id": request.right_tokenizer_id })),
            )
        }
    };

    let (left_backend, left_ids, left_hash) = {
        let mut left = left_handle.lock().await;
        let left_encoding = match encode_one(
            &mut left,
            &request.sequence,
            add_special_tokens,
            compare_build_flags,
        ) {
            Ok(v) => v,
            Err(msg) => {
                return json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "encode_failed",
                    &msg,
                    Some(json!({ "endpoint": "/v1/tokenizer/compare" })),
                )
            }
        };
        let ids = left_encoding.ids;
        let hash = sha256_ids(&ids);
        (left.backend.as_str().to_string(), ids, hash)
    };

    let (right_backend, right_ids, right_hash) = if Arc::ptr_eq(&left_handle, &right_handle) {
        (left_backend.clone(), left_ids.clone(), left_hash.clone())
    } else {
        let mut right = right_handle.lock().await;
        let right_encoding = match encode_one(
            &mut right,
            &request.sequence,
            add_special_tokens,
            compare_build_flags,
        ) {
            Ok(v) => v,
            Err(msg) => {
                return json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "encode_failed",
                    &msg,
                    Some(json!({ "endpoint": "/v1/tokenizer/compare" })),
                )
            }
        };
        let ids = right_encoding.ids;
        let hash = sha256_ids(&ids);
        (right.backend.as_str().to_string(), ids, hash)
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
                    "encode": ["sequence", "is_pretokenized", "add_special_tokens", "return", "include_hash", "benchmark", "accept:application/octet-stream(ids_only)"],
                    "encode_batch": ["inputs", "is_pretokenized", "add_special_tokens", "padding", "return", "include_hash", "benchmark", "accept:application/octet-stream(ids_only)"],
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
                    "encode": ["sequence", "is_pretokenized", "add_special_tokens", "return", "include_hash", "benchmark", "accept:application/octet-stream(ids_only)"],
                    "encode_batch": ["inputs", "is_pretokenized", "add_special_tokens", "padding", "return", "include_hash", "benchmark", "accept:application/octet-stream(ids_only)"],
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
                    "pair encoding"
                ],
                "talu": [
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

async fn find_tokenizer_instance(
    state: &AppState,
    tokenizer_id: &str,
) -> Option<TokenizerInstanceHandle> {
    let instances = state.tokenizer_instances.lock().await;
    instances.get(tokenizer_id).cloned()
}

fn validate_binary_ids_only_return_fields(
    return_fields: &ReturnFields,
    endpoint: &str,
) -> Result<(), Response<BoxBody>> {
    let valid = return_fields.include_ids()
        && !return_fields.include_tokens()
        && !return_fields.include_type_ids()
        && !return_fields.include_attention_mask()
        && !return_fields.include_special_tokens_mask()
        && !return_fields.include_offsets();

    if valid {
        return Ok(());
    }

    Err(json_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        "unsupported_option",
        "binary tokenizer responses only support return.ids=true with all other return fields false",
        Some(json!({
            "field": "return",
            "expected": {
                "ids": true,
                "tokens": false,
                "type_ids": false,
                "attention_mask": false,
                "special_tokens_mask": false,
                "offsets": false
            },
            "endpoint": endpoint
        })),
    ))
}

fn request_wants_binary_response(headers: &hyper::HeaderMap) -> bool {
    let Some(accept) = headers.get(hyper::header::ACCEPT) else {
        return false;
    };
    let Ok(accept) = accept.to_str() else {
        return false;
    };
    accept_header_allows_mime(accept, "application/octet-stream")
}

fn accept_header_allows_mime(accept_header: &str, target_mime: &str) -> bool {
    accept_header.split(',').any(|entry| {
        let mut parts = entry.trim().split(';');
        let Some(mime) = parts.next() else {
            return false;
        };
        if !mime.trim().eq_ignore_ascii_case(target_mime) {
            return false;
        }
        let q = parts
            .filter_map(parse_accept_q_parameter)
            .next_back()
            .unwrap_or(1.0);
        q > 0.0
    })
}

fn parse_accept_q_parameter(param: &str) -> Option<f32> {
    let (key, value) = param.split_once('=')?;
    if !key.trim().eq_ignore_ascii_case("q") {
        return None;
    }
    value.trim().parse::<f32>().ok()
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

fn normalize_input_sequence_with_mode(
    value: &Value,
    is_pretokenized: bool,
) -> Result<String, String> {
    if is_pretokenized {
        match value {
            Value::Array(parts) => {
                let mut tokens = Vec::with_capacity(parts.len());
                for part in parts {
                    let Some(token) = part.as_str() else {
                        return Err(
                            "sequence must be an array of strings when is_pretokenized=true"
                                .to_string(),
                        );
                    };
                    tokens.push(token);
                }
                Ok(tokens.join(" "))
            }
            _ => Err("sequence must be an array of strings when is_pretokenized=true".to_string()),
        }
    } else {
        match value {
            Value::String(s) => Ok(s.clone()),
            Value::Array(_) => {
                Err("sequence must be a string when is_pretokenized=false".to_string())
            }
            _ => Err("sequence must be a string when is_pretokenized=false".to_string()),
        }
    }
}

fn normalize_batch_inputs_with_mode(
    inputs: &[Value],
    is_pretokenized: bool,
) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(inputs.len());
    for item in inputs {
        out.push(normalize_input_sequence_with_mode(item, is_pretokenized)?);
    }
    Ok(out)
}

fn enforce_path_source_policy(path_value: &str) -> Result<(), Response<BoxBody>> {
    if !path_source_allowed() {
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "path_not_allowed",
            "source.kind=path is disabled by server policy; use source.kind=json or enable TALU_TOKENIZER_ALLOW_PATH_SOURCE=1 and TALU_TOKENIZER_ALLOWED_PATH_ROOTS",
            Some(json!({
                "field": "source.kind",
                "reason": "path_source_disabled",
                "expected": "source.kind=json or TALU_TOKENIZER_ALLOW_PATH_SOURCE=1",
                "endpoint": "/v1/tokenizer/instances"
            })),
        ));
    }

    let requested = std::fs::canonicalize(Path::new(path_value)).map_err(|e| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &format!("failed to canonicalize source.value path: {e}"),
            Some(json!({
                "field": "source.value",
                "reason": "canonicalize_failed",
                "got": path_value,
                "endpoint": "/v1/tokenizer/instances"
            })),
        )
    })?;

    let allowed_roots = tokenizer_path_allowlist_roots()?;
    if allowed_roots.is_empty() {
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "path_not_allowed",
            "source.kind=path requires TALU_TOKENIZER_ALLOWED_PATH_ROOTS",
            Some(json!({
                "field": "source.value",
                "reason": "allowlist_required",
                "expected": "TALU_TOKENIZER_ALLOWED_PATH_ROOTS must contain one or more canonical roots",
                "endpoint": "/v1/tokenizer/instances"
            })),
        ));
    }

    if !allowed_roots.iter().any(|root| requested.starts_with(root)) {
        let roots = allowed_roots
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "path_not_allowed",
            "source.value path is outside configured allowlist roots",
            Some(json!({
                "field": "source.value",
                "reason": "path_outside_allowlist",
                "expected": roots,
                "got": requested.to_string_lossy(),
                "endpoint": "/v1/tokenizer/instances"
            })),
        ));
    }

    Ok(())
}

fn path_source_allowed() -> bool {
    std::env::var("TALU_TOKENIZER_ALLOW_PATH_SOURCE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
        || std::env::var("TALU_TOKENIZER_ALLOW_PATH_SOURCE_FOR_AUTH")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
}

fn tokenizer_path_allowlist_roots() -> Result<Vec<std::path::PathBuf>, Response<BoxBody>> {
    tokenizer_path_allowlist_roots_raw().map_err(|(segment, err)| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_config_error",
            &format!("invalid TALU_TOKENIZER_ALLOWED_PATH_ROOTS entry `{segment}`: {err}"),
            Some(json!({
                "field": "TALU_TOKENIZER_ALLOWED_PATH_ROOTS",
                "reason": "invalid_allowlist_root",
                "got": segment,
                "endpoint": "/v1/tokenizer/instances"
            })),
        )
    })
}

fn tokenizer_path_allowlist_roots_raw() -> Result<Vec<std::path::PathBuf>, (String, String)> {
    let raw = std::env::var("TALU_TOKENIZER_ALLOWED_PATH_ROOTS").unwrap_or_default();
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut roots = Vec::new();
    for segment in raw.split(':').map(str::trim).filter(|s| !s.is_empty()) {
        let canonical = std::fs::canonicalize(Path::new(segment))
            .map_err(|e| (segment.to_string(), e.to_string()))?;
        roots.push(canonical);
    }

    Ok(roots)
}

pub fn validate_path_source_policy_env() -> Result<(), String> {
    if !path_source_allowed() {
        return Ok(());
    }

    let roots = tokenizer_path_allowlist_roots_raw().map_err(|(segment, err)| {
        format!("invalid TALU_TOKENIZER_ALLOWED_PATH_ROOTS entry `{segment}`: {err}")
    })?;

    if roots.is_empty() {
        return Err(
            "path-source policy enabled (TALU_TOKENIZER_ALLOW_PATH_SOURCE=1 or TALU_TOKENIZER_ALLOW_PATH_SOURCE_FOR_AUTH=1) requires TALU_TOKENIZER_ALLOWED_PATH_ROOTS".to_string(),
        );
    }

    Ok(())
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
    build_flags: EncodeBuildFlags,
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
    let attention_mask = if build_flags.include_attention_mask {
        copy_u32_slice(result.attention_mask, result.num_tokens)
    } else {
        Vec::new()
    };
    let special_tokens_mask = if build_flags.include_special_tokens_mask {
        copy_u32_slice(result.special_tokens_mask, result.num_tokens)
    } else {
        Vec::new()
    };
    let offsets = if build_flags.include_offsets {
        copy_offsets_slice(result.offsets, result.num_tokens)
    } else {
        Vec::new()
    };

    // SAFETY: result was returned by talu_tokenizer_encode and must be freed once.
    unsafe { talu_sys::talu_encode_result_free(result) };

    let tokens = if build_flags.include_tokens {
        ids_to_tokens(instance, &ids)?
    } else {
        Vec::new()
    };
    let type_ids = if build_flags.include_type_ids {
        vec![0u32; ids.len()]
    } else {
        Vec::new()
    };

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
            if !row.tokens.is_empty() {
                row.tokens
                    .extend(std::iter::repeat_n(config.pad_token.clone(), pad_count));
            }
            if !row.type_ids.is_empty() {
                row.type_ids
                    .extend(std::iter::repeat_n(config.pad_type_id, pad_count));
            }
            if !row.attention_mask.is_empty() {
                row.attention_mask.extend(std::iter::repeat_n(0, pad_count));
            }
            if !row.special_tokens_mask.is_empty() {
                row.special_tokens_mask
                    .extend(std::iter::repeat_n(1, pad_count));
            }
            if !row.offsets.is_empty() {
                row.offsets.extend(std::iter::repeat_n([0, 0], pad_count));
            }
        }
        PaddingDirection::Left => {
            prepend_repeat(&mut row.ids, config.pad_id, pad_count);
            if !row.tokens.is_empty() {
                prepend_repeat(&mut row.tokens, config.pad_token.clone(), pad_count);
            }
            if !row.type_ids.is_empty() {
                prepend_repeat(&mut row.type_ids, config.pad_type_id, pad_count);
            }
            if !row.attention_mask.is_empty() {
                prepend_repeat(&mut row.attention_mask, 0, pad_count);
            }
            if !row.special_tokens_mask.is_empty() {
                prepend_repeat(&mut row.special_tokens_mask, 1, pad_count);
            }
            if !row.offsets.is_empty() {
                prepend_repeat(&mut row.offsets, [0, 0], pad_count);
            }
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

fn encoding_payload_from_owned(
    row: MutableEncoding,
    return_fields: &ReturnFields,
) -> EncodingPayload {
    let MutableEncoding {
        ids,
        tokens,
        type_ids,
        attention_mask,
        special_tokens_mask,
        offsets,
    } = row;

    EncodingPayload {
        ids: return_fields.include_ids().then_some(ids),
        tokens: return_fields.include_tokens().then_some(tokens),
        type_ids: return_fields.include_type_ids().then_some(type_ids),
        attention_mask: return_fields
            .include_attention_mask()
            .then_some(attention_mask),
        special_tokens_mask: return_fields
            .include_special_tokens_mask()
            .then_some(special_tokens_mask),
        offsets: return_fields.include_offsets().then_some(offsets),
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

fn encode_ids_binary_v1(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + ids.len() * 4);
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

fn encode_batch_ids_binary_v1(rows: &[MutableEncoding]) -> Vec<u8> {
    let num_rows = rows.len() as u32;
    let total_ids = rows.iter().map(|row| row.ids.len()).sum::<usize>();
    let mut out = Vec::with_capacity(4 + rows.len() * 4 + total_ids * 4);
    out.extend_from_slice(&num_rows.to_le_bytes());
    for row in rows {
        out.extend_from_slice(&(row.ids.len() as u32).to_le_bytes());
    }
    for row in rows {
        for &id in &row.ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
    out
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
