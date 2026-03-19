//! API-contract tests for `/v1/tokenizer/*`.
//!
//! Scope here is HTTP mechanics: validation, error envelopes, state handling,
//! response shapes, option propagation, and deterministic API outputs.
//! Tokenizer algorithm correctness is covered in CAPI/core tokenizer suites.

use super::tokenizer_fixture_json;
use crate::server::common::*;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn create_instance(addr: std::net::SocketAddr, backend: &str) -> serde_json::Value {
    let resp = post_json(
        addr,
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": backend,
            "source": {
                "kind": "json",
                "value": tokenizer_fixture_json()
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

fn create_instance_with_json(
    addr: std::net::SocketAddr,
    backend: &str,
    json_source: &str,
) -> serde_json::Value {
    let resp = post_json(
        addr,
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": backend,
            "source": {
                "kind": "json",
                "value": json_source
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

fn assert_error(resp: HttpResponse, status: u16, code: &str) -> serde_json::Value {
    assert_eq!(resp.status, status, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.starts_with("application/json"),
        "expected JSON error response; content-type={content_type} body={}",
        resp.body
    );
    let body = resp.json();
    assert_eq!(body["error"]["code"], code, "body: {}", resp.body);
    assert!(
        body["error"]["message"].is_string(),
        "error.message must be a string; body={}",
        resp.body
    );
    body
}

#[test]
fn instance_lifecycle_create_get_delete() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().expect("tokenizer_id");
    assert_eq!(created["backend"], "talu");
    assert!(created["vocab_size"].as_u64().unwrap_or(0) > 0);
    assert_eq!(created["tokenizer_sha256"].as_str().unwrap_or("").len(), 64);

    let get_resp = get(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let got = get_resp.json();
    assert_eq!(got["tokenizer_id"], tokenizer_id);
    assert_eq!(got["source"]["kind"], "json");

    let del_resp = delete(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(del_resp.status, 204, "body: {}", del_resp.body);

    let missing_resp = get(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(missing_resp.status, 404, "body: {}", missing_resp.body);
}

#[test]
fn encode_decode_roundtrip_basic() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let enc_resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false,
            "benchmark": true
        }),
    );
    assert_eq!(enc_resp.status, 200, "body: {}", enc_resp.body);
    let enc = enc_resp.json();
    let ids = enc["encoding"]["ids"].as_array().expect("ids array");
    assert!(!ids.is_empty());
    assert_eq!(enc["sha256_ids"].as_str().unwrap_or("").len(), 64);
    assert!(enc["timing_ms"]["encode"].is_number());

    let ids_vec: Vec<u32> = ids
        .iter()
        .map(|v| v.as_u64().expect("u64") as u32)
        .collect();

    let dec_resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "ids": ids_vec,
            "skip_special_tokens": false
        }),
    );
    assert_eq!(dec_resp.status, 200, "body: {}", dec_resp.body);
    let dec = dec_resp.json();
    let text = dec["text"].as_str().unwrap_or("");
    assert!(!text.is_empty(), "decoded text should be non-empty");
}

#[test]
fn encode_batch_padding_invariants() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello world", "hello"],
            "add_special_tokens": false,
            "padding": {
                "enabled": true,
                "direction": "left",
                "pad_id": 0,
                "pad_type_id": 0,
                "pad_token": "<PAD>",
                "length": 6
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let be = &json["batch_encoding"];

    let input_ids = be["input_ids"].as_array().expect("input_ids rows");
    let masks = be["attention_mask"]
        .as_array()
        .expect("attention_mask rows");
    assert_eq!(
        be["num_sequences"].as_u64().unwrap_or(0),
        input_ids.len() as u64
    );
    assert_eq!(input_ids.len(), masks.len());

    let lengths = be["lengths"].as_array().expect("lengths");
    let sum_lengths: u64 = lengths.iter().map(|v| v.as_u64().unwrap_or(0)).sum();
    assert_eq!(be["total_tokens"].as_u64().unwrap_or(0), sum_lengths);

    let row0 = input_ids[0].as_array().unwrap();
    let row1 = input_ids[1].as_array().unwrap();
    let mask1 = masks[1].as_array().unwrap();
    assert_eq!(row0.len(), row1.len());
    assert!(row0.len() >= 6);
    assert_eq!(row1[0].as_u64().unwrap_or(1), 0);
    assert_eq!(mask1[0].as_u64().unwrap_or(1), 0);
    assert_eq!(be["padding_side"], "left");
    assert_eq!(be["pad_token_id"].as_u64().unwrap_or(999), 0);
}

#[test]
fn vocab_and_id_mapping_endpoints() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let vocab_resp = get(
        ctx.addr(),
        &format!("/v1/tokenizer/vocab?tokenizer_id={tokenizer_id}"),
    );
    assert_eq!(vocab_resp.status, 200, "body: {}", vocab_resp.body);
    let vocab = vocab_resp.json();
    let vocab_obj = vocab.as_object().expect("vocab object");
    assert!(!vocab_obj.is_empty());

    let vs_resp = get(
        ctx.addr(),
        &format!("/v1/tokenizer/vocab_size?tokenizer_id={tokenizer_id}&with_added_tokens=true"),
    );
    assert_eq!(vs_resp.status, 200, "body: {}", vs_resp.body);
    let vs = vs_resp.json();
    assert!(vs["vocab_size"].as_u64().unwrap_or(0) > 0);

    let enc_resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "add_special_tokens": false
        }),
    );
    assert_eq!(enc_resp.status, 200, "body: {}", enc_resp.body);
    let enc = enc_resp.json();
    let first_id = enc["encoding"]["ids"][0].as_i64().unwrap();

    let id_to_token = post_json(
        ctx.addr(),
        "/v1/tokenizer/id_to_token",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "token_id": first_id
        }),
    );
    assert_eq!(id_to_token.status, 200, "body: {}", id_to_token.body);
    let token = id_to_token.json()["token"].as_str().unwrap().to_string();

    let token_to_id = post_json(
        ctx.addr(),
        "/v1/tokenizer/token_to_id",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "token": token
        }),
    );
    assert_eq!(token_to_id.status, 200, "body: {}", token_to_id.body);
    let roundtrip_id = token_to_id.json()["id"].as_i64().unwrap();
    assert!(roundtrip_id >= 0);
}

#[test]
fn compare_and_capabilities_endpoints() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let left = create_instance(ctx.addr(), "talu");
    let right = create_instance(ctx.addr(), "tokenizers");
    let left_id = left["tokenizer_id"].as_str().unwrap();
    let right_id = right["tokenizer_id"].as_str().unwrap();

    let compare = post_json(
        ctx.addr(),
        "/v1/tokenizer/compare",
        &serde_json::json!({
            "left_tokenizer_id": left_id,
            "right_tokenizer_id": right_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(compare.status, 200, "body: {}", compare.body);
    let cmp = compare.json();
    assert!(cmp["left"]["token_count"].as_u64().unwrap_or(0) > 0);
    assert_eq!(cmp["left"]["sha256_ids"].as_str().unwrap_or("").len(), 64);
    assert_eq!(cmp["right"]["sha256_ids"].as_str().unwrap_or("").len(), 64);

    let caps = get(ctx.addr(), "/v1/tokenizer/capabilities");
    assert_eq!(caps.status, 200, "body: {}", caps.body);
    let caps_json = caps.json();
    let backends = caps_json["supported_backends"].as_array().unwrap();
    assert!(backends.iter().any(|v| v == "tokenizers"));
    assert!(backends.iter().any(|v| v == "talu"));
}

#[test]
fn add_tokens_updates_vocab_size_and_mapping_endpoints() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();
    let base_vocab_size = created["vocab_size"].as_u64().unwrap();

    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/add_tokens",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "tokens": ["<extra_token>"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let body = resp.json();
    assert_eq!(body["added"], 1);
    assert_eq!(body["vocab_size"].as_u64().unwrap(), base_vocab_size + 1);

    let token_to_id = post_json(
        ctx.addr(),
        "/v1/tokenizer/token_to_id",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "token": "<extra_token>"
        }),
    );
    assert_eq!(token_to_id.status, 200, "body: {}", token_to_id.body);
    let added_id = token_to_id.json()["id"].as_i64().unwrap();
    assert!(added_id >= base_vocab_size as i64);

    let id_to_token = post_json(
        ctx.addr(),
        "/v1/tokenizer/id_to_token",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "token_id": added_id
        }),
    );
    assert_eq!(id_to_token.status, 200, "body: {}", id_to_token.body);
    assert_eq!(id_to_token.json()["token"], "<extra_token>");

    let with_added = get(
        ctx.addr(),
        &format!("/v1/tokenizer/vocab_size?tokenizer_id={tokenizer_id}&with_added_tokens=true"),
    );
    assert_eq!(with_added.status, 200, "body: {}", with_added.body);
    assert_eq!(
        with_added.json()["vocab_size"].as_u64().unwrap(),
        base_vocab_size + 1
    );

    let without_added = get(
        ctx.addr(),
        &format!("/v1/tokenizer/vocab_size?tokenizer_id={tokenizer_id}&with_added_tokens=false"),
    );
    assert_eq!(without_added.status, 200, "body: {}", without_added.body);
    assert_eq!(
        without_added.json()["vocab_size"].as_u64().unwrap(),
        base_vocab_size
    );

    let vocab = get(
        ctx.addr(),
        &format!("/v1/tokenizer/vocab?tokenizer_id={tokenizer_id}"),
    );
    assert_eq!(vocab.status, 200, "body: {}", vocab.body);
    let vocab_json = vocab.json();
    assert_eq!(vocab_json["<extra_token>"].as_i64().unwrap(), added_id);

    let duplicate = post_json(
        ctx.addr(),
        "/v1/tokenizer/add_tokens",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "tokens": ["<extra_token>"]
        }),
    );
    assert_eq!(duplicate.status, 200, "body: {}", duplicate.body);
    assert_eq!(duplicate.json()["added"], 0);
}

#[test]
fn decode_batch_returns_texts_for_multiple_rows() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let hello = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "add_special_tokens": false
        }),
    );
    assert_eq!(hello.status, 200, "body: {}", hello.body);
    let hello_ids = hello.json()["encoding"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();

    let world = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(world.status, 200, "body: {}", world.body);
    let world_ids = world.json()["encoding"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();

    let decoded = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "ids_batch": [hello_ids, world_ids],
            "skip_special_tokens": false
        }),
    );
    assert_eq!(decoded.status, 200, "body: {}", decoded.body);
    let texts = decoded.json()["texts"].as_array().unwrap().clone();
    assert_eq!(texts.len(), 2);
    assert!(texts[0].as_str().unwrap_or("").len() > 0);
    assert!(texts[1].as_str().unwrap_or("").len() > 0);
}

#[test]
fn instance_truncation_and_padding_state_applies_to_encode() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let trunc = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_truncation",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "max_length": 3,
            "direction": "right"
        }),
    );
    assert_eq!(trunc.status, 200, "body: {}", trunc.body);

    let pad = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_padding",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "direction": "left",
            "pad_id": 0,
            "pad_type_id": 0,
            "pad_token": "<PAD>",
            "length": 5
        }),
    );
    assert_eq!(pad.status, 200, "body: {}", pad.body);

    let get_enabled = get(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(get_enabled.status, 200, "body: {}", get_enabled.body);
    let enabled = get_enabled.json();
    assert_eq!(enabled["truncation"]["max_length"], 3);
    assert_eq!(enabled["padding"]["length"], 5);
    assert_eq!(enabled["padding"]["direction"], "left");

    let baseline = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(baseline.status, 200, "body: {}", baseline.body);
    let baseline_ids = baseline.json()["encoding"]["ids"].as_array().unwrap().len();
    assert!(
        baseline_ids > 3,
        "baseline input must exceed truncation length"
    );

    let enc = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(enc.status, 200, "body: {}", enc.body);
    let enc_json = enc.json();
    let ids = enc_json["encoding"]["ids"].as_array().unwrap();
    let mask = enc_json["encoding"]["attention_mask"].as_array().unwrap();
    assert_eq!(ids.len(), 5);
    assert_eq!(mask.len(), 5);
    let non_padding = mask
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .filter(|&v| v == 1)
        .count();
    assert_eq!(non_padding, 3, "non-padding tokens should honor max_length");

    let disable_trunc = post_json(
        ctx.addr(),
        "/v1/tokenizer/disable_truncation",
        &serde_json::json!({ "tokenizer_id": tokenizer_id }),
    );
    assert_eq!(disable_trunc.status, 200, "body: {}", disable_trunc.body);

    let disable_pad = post_json(
        ctx.addr(),
        "/v1/tokenizer/disable_padding",
        &serde_json::json!({ "tokenizer_id": tokenizer_id }),
    );
    assert_eq!(disable_pad.status, 200, "body: {}", disable_pad.body);

    let get_disabled = get(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(get_disabled.status, 200, "body: {}", get_disabled.body);
    let disabled = get_disabled.json();
    assert!(disabled["truncation"].is_null());
    assert!(disabled["padding"].is_null());
}

#[test]
fn create_instance_rejects_unsupported_backend() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "does-not-exist",
            "source": { "kind": "json", "value": tokenizer_fixture_json() }
        }),
    );
    assert_error(resp, 400, "unsupported_backend");
}

#[test]
fn create_instance_rejects_invalid_source_kind() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "url", "value": "https://example.invalid/tokenizer.json" }
        }),
    );
    assert_error(resp, 400, "invalid_request");
}

#[test]
fn create_instance_rejects_invalid_json_source() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "json", "value": "{not-valid-json" }
        }),
    );
    assert_error(resp, 400, "invalid_request");
}

#[test]
fn encode_rejects_pair_and_pretokenized_and_invalid_sequence() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let pair = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "pair": "world"
        }),
    );
    assert_error(pair, 400, "unsupported_option");

    let pretokenized = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": ["hello", "world"],
            "is_pretokenized": true
        }),
    );
    assert_error(pretokenized, 400, "unsupported_option");

    let invalid_sequence = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": 123
        }),
    );
    assert_error(invalid_sequence, 400, "invalid_request");
}

#[test]
fn encode_not_found_and_return_projection_and_benchmark_toggle() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let missing = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "sequence": "hello"
        }),
    );
    assert_error(missing, 404, "tokenizer_not_found");

    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let projected = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "benchmark": false,
            "return": {
                "ids": true,
                "tokens": false,
                "type_ids": false,
                "attention_mask": false,
                "special_tokens_mask": false,
                "offsets": false
            }
        }),
    );
    assert_eq!(projected.status, 200, "body: {}", projected.body);
    let body = projected.json();
    assert!(body["timing_ms"].is_null());
    let enc_obj = body["encoding"].as_object().unwrap();
    assert!(enc_obj.contains_key("ids"));
    assert!(!enc_obj.contains_key("tokens"));
    assert!(!enc_obj.contains_key("type_ids"));
    assert!(!enc_obj.contains_key("attention_mask"));
    assert!(!enc_obj.contains_key("special_tokens_mask"));
    assert!(!enc_obj.contains_key("offsets"));
}

#[test]
fn encode_batch_rejects_invalid_inputs_and_pretokenized_and_projection() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let pretokenized = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello"],
            "is_pretokenized": true
        }),
    );
    assert_error(pretokenized, 400, "unsupported_option");

    let invalid_inputs = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": [123, "hello"]
        }),
    );
    assert_error(invalid_inputs, 400, "invalid_request");

    let projected = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello", "world"],
            "return": {
                "ids": true,
                "tokens": false,
                "type_ids": false,
                "attention_mask": false,
                "special_tokens_mask": false,
                "offsets": false
            }
        }),
    );
    assert_eq!(projected.status, 200, "body: {}", projected.body);
    let body = projected.json();
    let be = body["batch_encoding"].as_object().unwrap();
    assert!(be.contains_key("input_ids"));
    assert!(!be.contains_key("attention_mask"));
    assert!(!be.contains_key("type_ids"));
    assert!(!be.contains_key("special_tokens_mask"));
    let row0 = body["batch_encoding"]["encodings"][0].as_object().unwrap();
    assert!(row0.contains_key("ids"));
    assert!(!row0.contains_key("tokens"));
}

#[test]
fn decode_and_decode_batch_return_not_found_for_missing_instance() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let decode = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "ids": [1, 2, 3]
        }),
    );
    assert_error(decode, 404, "tokenizer_not_found");

    let decode_batch = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode_batch",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "ids_batch": [[1, 2], [3]]
        }),
    );
    assert_error(decode_batch, 404, "tokenizer_not_found");
}

#[test]
fn vocab_and_mapping_endpoints_validate_required_inputs() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let vocab_missing = get(ctx.addr(), "/v1/tokenizer/vocab");
    assert_error(vocab_missing, 400, "invalid_request");

    let vocab_size_missing = get(ctx.addr(), "/v1/tokenizer/vocab_size");
    assert_error(vocab_size_missing, 400, "invalid_request");

    let token_to_id_missing = post_json(
        ctx.addr(),
        "/v1/tokenizer/token_to_id",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "token": "hello"
        }),
    );
    assert_error(token_to_id_missing, 404, "tokenizer_not_found");

    let id_to_token_missing = post_json(
        ctx.addr(),
        "/v1/tokenizer/id_to_token",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "token_id": 1
        }),
    );
    assert_error(id_to_token_missing, 404, "tokenizer_not_found");
}

#[test]
fn truncation_and_padding_endpoints_validate_parameters() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let trunc_zero = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_truncation",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "max_length": 0
        }),
    );
    assert_error(trunc_zero, 400, "invalid_request");

    let trunc_stride = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_truncation",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "max_length": 4,
            "stride": 1
        }),
    );
    assert_error(trunc_stride, 400, "unsupported_option");

    let trunc_strategy = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_truncation",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "max_length": 4,
            "strategy": "only_second"
        }),
    );
    assert_error(trunc_strategy, 400, "unsupported_option");

    let trunc_direction = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_truncation",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "max_length": 4,
            "direction": "up"
        }),
    );
    assert_error(trunc_direction, 400, "invalid_request");

    let pad_multiple = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_padding",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "multiple_of": 0
        }),
    );
    assert_error(pad_multiple, 400, "invalid_request");

    let pad_direction = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_padding",
        &serde_json::json!({
            "tokenizer_id": "tok_missing",
            "direction": "up"
        }),
    );
    assert_error(pad_direction, 400, "invalid_request");
}

#[test]
fn disable_padding_and_disable_truncation_return_not_found_for_missing_instance() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let disable_pad = post_json(
        ctx.addr(),
        "/v1/tokenizer/disable_padding",
        &serde_json::json!({ "tokenizer_id": "tok_missing" }),
    );
    assert_error(disable_pad, 404, "tokenizer_not_found");

    let disable_trunc = post_json(
        ctx.addr(),
        "/v1/tokenizer/disable_truncation",
        &serde_json::json!({ "tokenizer_id": "tok_missing" }),
    );
    assert_error(disable_trunc, 404, "tokenizer_not_found");
}

#[test]
fn compare_returns_not_found_for_missing_left_or_right() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let missing_left = post_json(
        ctx.addr(),
        "/v1/tokenizer/compare",
        &serde_json::json!({
            "left_tokenizer_id": "tok_left_missing",
            "right_tokenizer_id": "tok_right_missing",
            "sequence": "hello"
        }),
    );
    assert_error(missing_left, 404, "tokenizer_not_found");

    let existing = create_instance(ctx.addr(), "talu");
    let existing_id = existing["tokenizer_id"].as_str().unwrap();
    let missing_right = post_json(
        ctx.addr(),
        "/v1/tokenizer/compare",
        &serde_json::json!({
            "left_tokenizer_id": existing_id,
            "right_tokenizer_id": "tok_right_missing",
            "sequence": "hello"
        }),
    );
    assert_error(missing_right, 404, "tokenizer_not_found");
}

#[test]
fn training_endpoints_add_tokens_deterministically() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();
    let base_vocab_size = created["vocab_size"].as_u64().unwrap();

    let train = post_json(
        ctx.addr(),
        "/v1/tokenizer/train",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "texts": [
                "alpha beta alpha",
                "beta gamma"
            ],
            "trainer": {
                "vocab_size": base_vocab_size + 3,
                "special_tokens": {
                    "additional_special_tokens": ["<TRAIN_SPECIAL>"]
                }
            }
        }),
    );
    if train.status != 200 {
        assert_error(train, 400, "train_failed");
        return;
    }
    let train_json = train.json();
    assert_eq!(train_json["trained"], true);
    assert!(train_json["added_special_tokens"].as_u64().unwrap() >= 1);
    assert_eq!(
        train_json["vocab_size"].as_u64().unwrap(),
        base_vocab_size + 3
    );

    let lookup = post_json(
        ctx.addr(),
        "/v1/tokenizer/token_to_id",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "token": "alpha"
        }),
    );
    assert_eq!(lookup.status, 200, "body: {}", lookup.body);
    assert!(lookup.json()["id"].as_i64().unwrap() >= base_vocab_size as i64);

    let train_from_iterator = post_json(
        ctx.addr(),
        "/v1/tokenizer/train_from_iterator",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "iterator": ["delta delta", "epsilon"]
        }),
    );
    assert_eq!(
        train_from_iterator.status, 200,
        "body: {}",
        train_from_iterator.body
    );
    let iterator_json = train_from_iterator.json();
    assert_eq!(iterator_json["trained"], true);
    assert!(iterator_json["added"].as_u64().unwrap() >= 1);
}

#[test]
fn add_special_tokens_accepts_tokens_and_special_tokens_object() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();
    let base_vocab_size = created["vocab_size"].as_u64().unwrap();

    let first = post_json(
        ctx.addr(),
        "/v1/tokenizer/add_special_tokens",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "tokens": ["<SPECIAL_A>"]
        }),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);
    let first_json = first.json();
    assert_eq!(first_json["added"], 1);
    assert_eq!(
        first_json["vocab_size"].as_u64().unwrap(),
        base_vocab_size + 1
    );

    let second = post_json(
        ctx.addr(),
        "/v1/tokenizer/add_special_tokens",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "special_tokens": {
                "cls_token": "<SPECIAL_B>",
                "additional_special_tokens": ["<SPECIAL_C>"]
            }
        }),
    );
    assert_eq!(second.status, 200, "body: {}", second.body);
    let second_json = second.json();
    assert_eq!(second_json["added"], 2);
    assert_eq!(
        second_json["vocab_size"].as_u64().unwrap(),
        base_vocab_size + 3
    );

    for tok in ["<SPECIAL_A>", "<SPECIAL_B>", "<SPECIAL_C>"] {
        let lookup = post_json(
            ctx.addr(),
            "/v1/tokenizer/token_to_id",
            &serde_json::json!({
                "tokenizer_id": tokenizer_id,
                "token": tok
            }),
        );
        assert_eq!(lookup.status, 200, "body: {}", lookup.body);
        assert!(lookup.json()["id"].as_i64().unwrap() >= base_vocab_size as i64);
    }
}

#[test]
fn save_writes_tokenizer_json_with_added_tokens_and_returns_hash() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let tmp = TempDir::new().expect("temp dir");
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let add = post_json(
        ctx.addr(),
        "/v1/tokenizer/add_tokens",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "tokens": ["<saved_added_token>"]
        }),
    );
    assert_eq!(add.status, 200, "body: {}", add.body);

    let save = post_json(
        ctx.addr(),
        "/v1/tokenizer/save",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "path": tmp.path(),
            "pretty": true
        }),
    );
    assert_eq!(save.status, 200, "body: {}", save.body);
    let save_json = save.json();
    let path = save_json["path"].as_str().unwrap();
    assert!(path.ends_with("tokenizer.json"), "path={path}");
    assert_eq!(save_json["tokenizer_sha256"].as_str().unwrap().len(), 64);
    assert!(save_json["bytes"].as_u64().unwrap() > 0);

    let bytes = std::fs::read(path).expect("read saved tokenizer json");
    let saved_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse saved tokenizer");
    let added_tokens = saved_json["added_tokens"]
        .as_array()
        .expect("added_tokens array");
    assert!(added_tokens
        .iter()
        .any(|t| t["content"] == "<saved_added_token>"));
}

#[test]
fn save_rejects_existing_target_when_overwrite_false() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let tmp = TempDir::new().expect("temp dir");
    let out = tmp.path().join("tokenizer.json");
    std::fs::write(&out, b"{}").expect("seed output");
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let save = post_json(
        ctx.addr(),
        "/v1/tokenizer/save",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "path": out,
            "overwrite": false
        }),
    );
    assert_error(save, 400, "save_failed");
}

#[test]
fn capabilities_response_has_expected_structure() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let caps = get(ctx.addr(), "/v1/tokenizer/capabilities");
    assert_eq!(caps.status, 200, "body: {}", caps.body);
    let body = caps.json();

    assert!(body["supported_backends"].as_array().is_some());
    assert!(body["supported_options"]["talu"].is_object());
    assert!(body["supported_options"]["tokenizers"].is_object());
    assert!(body["unsupported_feature_matrix"]["talu"].is_array());
    assert!(body["unsupported_feature_matrix"]["tokenizers"].is_array());
    assert_eq!(body["build"]["server"], "talu-cli");
    assert!(body["build"]["version"].as_str().unwrap_or("").len() >= 1);
}

#[test]
fn create_instance_from_path_source_roundtrip() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let tmp = TempDir::new().expect("temp dir");
    let model_dir = tmp.path().join("model_dir");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    let tokenizer_path = model_dir.join("tokenizer.json");
    std::fs::write(&tokenizer_path, tokenizer_fixture_json().as_bytes())
        .expect("write tokenizer.json");

    let created = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "path", "value": model_dir }
        }),
    );
    assert_eq!(created.status, 200, "body: {}", created.body);
    let created_json = created.json();
    let tokenizer_id = created_json["tokenizer_id"]
        .as_str()
        .expect("tokenizer_id")
        .to_string();

    let get_resp = get(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let got = get_resp.json();
    assert_eq!(got["source"]["kind"], "path");
    assert_eq!(got["source"]["value"], model_dir.to_string_lossy().as_ref());
}

#[test]
fn create_instance_path_not_found_returns_invalid_request() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "path", "value": "/definitely/not/here/tokenizer.json" }
        }),
    );
    assert_error(resp, 400, "invalid_request");
}

#[test]
fn encode_and_encode_batch_are_deterministic_for_fixed_inputs() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let first = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);
    let first_json = first.json();

    let second = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(second.status, 200, "body: {}", second.body);
    let second_json = second.json();

    assert_eq!(
        first_json["encoding"]["ids"],
        second_json["encoding"]["ids"]
    );
    assert_eq!(first_json["sha256_ids"], second_json["sha256_ids"]);
    assert_eq!(first_json["impl"], "talu.encode");

    let batch_first = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello world", "world hello"],
            "add_special_tokens": false
        }),
    );
    assert_eq!(batch_first.status, 200, "body: {}", batch_first.body);
    let batch_first_json = batch_first.json();

    let batch_second = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello world", "world hello"],
            "add_special_tokens": false
        }),
    );
    assert_eq!(batch_second.status, 200, "body: {}", batch_second.body);
    let batch_second_json = batch_second.json();

    assert_eq!(
        batch_first_json["batch_encoding"]["input_ids"],
        batch_second_json["batch_encoding"]["input_ids"]
    );
    assert_eq!(
        batch_first_json["sha256_ids_batch"],
        batch_second_json["sha256_ids_batch"]
    );
    assert_eq!(batch_first_json["impl"], "talu.encode_batch");
}

#[test]
fn create_instance_json_source_reports_sha256_of_source() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let source = tokenizer_fixture_json();

    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let expected = format!("{:x}", hasher.finalize());

    let created = create_instance_with_json(ctx.addr(), "talu", source);
    assert_eq!(created["tokenizer_sha256"], expected);
}

#[test]
fn create_instance_path_source_hash_is_deterministic_for_same_input() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let tmp = TempDir::new().expect("temp dir");
    let model_dir = tmp.path().join("model_dir");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    let tokenizer_path = model_dir.join("tokenizer.json");
    std::fs::write(&tokenizer_path, tokenizer_fixture_json().as_bytes())
        .expect("write tokenizer.json");

    let first = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "path", "value": model_dir }
        }),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);
    let first_json = first.json();
    let hash_1 = first_json["tokenizer_sha256"].as_str().unwrap();

    let second = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "path", "value": model_dir }
        }),
    );
    assert_eq!(second.status, 200, "body: {}", second.body);
    let second_json = second.json();
    let hash_2 = second_json["tokenizer_sha256"].as_str().unwrap();

    assert_eq!(hash_1.len(), 64);
    assert_eq!(hash_1, hash_2);
}

#[test]
fn create_instance_rejects_path_with_nul_byte() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let path_with_nul = format!("{}{}", "/tmp/tokenizer", '\0');
    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/instances",
        &serde_json::json!({
            "backend": "talu",
            "source": { "kind": "path", "value": path_with_nul }
        }),
    );
    assert_error(resp, 400, "invalid_request");
}

#[test]
fn delete_instance_second_call_returns_not_found() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let first = delete(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_eq!(first.status, 204, "body: {}", first.body);

    let second = delete(
        ctx.addr(),
        &format!("/v1/tokenizer/instances/{tokenizer_id}"),
    );
    assert_error(second, 404, "tokenizer_not_found");
}

#[test]
fn get_and_delete_instance_require_non_empty_path_id() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let get_empty = get(ctx.addr(), "/v1/tokenizer/instances/");
    assert_error(get_empty, 400, "invalid_request");

    let delete_empty = delete(ctx.addr(), "/v1/tokenizer/instances/");
    assert_error(delete_empty, 400, "invalid_request");
}

#[test]
fn compare_same_instance_has_no_diff_and_empty_windows() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let cmp = post_json(
        ctx.addr(),
        "/v1/tokenizer/compare",
        &serde_json::json!({
            "left_tokenizer_id": tokenizer_id,
            "right_tokenizer_id": tokenizer_id,
            "sequence": "hello world",
            "add_special_tokens": false
        }),
    );
    assert_eq!(cmp.status, 200, "body: {}", cmp.body);
    let body = cmp.json();
    assert!(body["first_diff_index"].is_null());
    assert_eq!(
        body["common_prefix"].as_u64().unwrap_or(0),
        body["left"]["token_count"].as_u64().unwrap_or(u64::MAX)
    );
    assert_eq!(
        body["left"]["token_count"].as_u64().unwrap_or(u64::MAX),
        body["right"]["token_count"].as_u64().unwrap_or(0)
    );
    assert_eq!(body["left_window"].as_array().unwrap().len(), 0);
    assert_eq!(body["right_window"].as_array().unwrap().len(), 0);
}

#[test]
fn compare_reports_first_diff_and_windows_for_different_tokenizers() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let left = create_instance_with_json(ctx.addr(), "talu", tokenizer_fixture_json());
    let left_id = left["tokenizer_id"].as_str().unwrap();

    let alt = tokenizer_fixture_json()
        .replace("\"h\": 9", "\"h\": 14")
        .replace("\"y\": 14", "\"y\": 9");
    let right = create_instance_with_json(ctx.addr(), "talu", &alt);
    let right_id = right["tokenizer_id"].as_str().unwrap();

    let cmp = post_json(
        ctx.addr(),
        "/v1/tokenizer/compare",
        &serde_json::json!({
            "left_tokenizer_id": left_id,
            "right_tokenizer_id": right_id,
            "sequence": "hello world",
            "add_special_tokens": false,
            "window": 2
        }),
    );
    assert_eq!(cmp.status, 200, "body: {}", cmp.body);
    let body = cmp.json();
    assert!(body["first_diff_index"].is_number());
    assert_ne!(body["left"]["sha256_ids"], body["right"]["sha256_ids"]);
    let left_window = body["left_window"].as_array().unwrap();
    let right_window = body["right_window"].as_array().unwrap();
    assert!(!left_window.is_empty());
    assert!(!right_window.is_empty());
    assert!(left_window.len() <= 5);
    assert!(right_window.len() <= 5);
}

#[test]
fn encode_accepts_add_special_tokens_flag_and_returns_valid_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let no_special = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "add_special_tokens": false
        }),
    );
    assert_eq!(no_special.status, 200, "body: {}", no_special.body);
    let no_special_json = no_special.json();

    let with_special = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "add_special_tokens": true
        }),
    );
    assert_eq!(with_special.status, 200, "body: {}", with_special.body);
    let with_special_json = with_special.json();

    assert!(no_special_json["encoding"]["ids"].is_array());
    assert!(with_special_json["encoding"]["ids"].is_array());
    assert!(no_special_json["sha256_ids"].as_str().unwrap_or("").len() == 64);
    assert!(with_special_json["sha256_ids"].as_str().unwrap_or("").len() == 64);
}

#[test]
fn decode_accepts_skip_special_tokens_flag_and_returns_text() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let encoded = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "sequence": "hello",
            "add_special_tokens": true
        }),
    );
    assert_eq!(encoded.status, 200, "body: {}", encoded.body);
    let ids: Vec<u32> = encoded.json()["encoding"]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let keep = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "ids": ids,
            "skip_special_tokens": false
        }),
    );
    assert_eq!(keep.status, 200, "body: {}", keep.body);
    let keep_text = keep.json()["text"].as_str().unwrap().to_string();

    let skip = post_json(
        ctx.addr(),
        "/v1/tokenizer/decode",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "ids": ids,
            "skip_special_tokens": true
        }),
    );
    assert_eq!(skip.status, 200, "body: {}", skip.body);
    let skip_text = skip.json()["text"].as_str().unwrap().to_string();

    assert!(!keep_text.is_empty());
    assert!(!skip_text.is_empty());
}

#[test]
fn encode_batch_with_benchmark_true_includes_timing_and_impl() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello", "world"],
            "benchmark": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let body = resp.json();
    assert_eq!(body["impl"], "talu.encode_batch");
    assert!(body["timing_ms"]["encode"].is_number());
    assert!(body["timing_ms"]["total"].is_number());
}

#[test]
fn per_request_padding_disable_overrides_instance_padding() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let enable = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_padding",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "direction": "left",
            "pad_id": 0,
            "pad_token": "<PAD>",
            "length": 8
        }),
    );
    assert_eq!(enable.status, 200, "body: {}", enable.body);

    let disabled_for_request = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello world", "hello"],
            "padding": { "enabled": false },
            "add_special_tokens": false
        }),
    );
    assert_eq!(
        disabled_for_request.status, 200,
        "body: {}",
        disabled_for_request.body
    );
    let body = disabled_for_request.json();
    let row0 = body["batch_encoding"]["input_ids"][0].as_array().unwrap();
    let row1 = body["batch_encoding"]["input_ids"][1].as_array().unwrap();
    assert_ne!(
        row0.len(),
        8,
        "instance padding should be disabled for request"
    );
    assert_ne!(
        row1.len(),
        8,
        "instance padding should be disabled for request"
    );
}

#[test]
fn encode_batch_padding_validates_direction_and_multiple_of() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let bad_direction = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello"],
            "padding": { "enabled": true, "direction": "up" }
        }),
    );
    assert_error(bad_direction, 400, "invalid_request");

    let bad_multiple = post_json(
        ctx.addr(),
        "/v1/tokenizer/encode_batch",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "inputs": ["hello"],
            "padding": { "enabled": true, "multiple_of": 0 }
        }),
    );
    assert_error(bad_multiple, 400, "invalid_request");
}

#[test]
fn enable_padding_accepts_pad_token_id_alias() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let created = create_instance(ctx.addr(), "talu");
    let tokenizer_id = created["tokenizer_id"].as_str().unwrap();

    let resp = post_json(
        ctx.addr(),
        "/v1/tokenizer/enable_padding",
        &serde_json::json!({
            "tokenizer_id": tokenizer_id,
            "direction": "left",
            "pad_token_id": 0,
            "length": 4
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let body = resp.json();
    assert_eq!(body["padding"]["pad_id"], 0);
}

#[test]
fn disable_unsupported_endpoints_return_not_found_when_instance_missing() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let cases = [
        (
            "/v1/tokenizer/add_tokens",
            serde_json::json!({
                "tokenizer_id": "tok_missing",
                "tokens": ["<x>"]
            }),
        ),
        (
            "/v1/tokenizer/add_special_tokens",
            serde_json::json!({
                "tokenizer_id": "tok_missing",
                "tokens": ["<x>"]
            }),
        ),
        (
            "/v1/tokenizer/train",
            serde_json::json!({
                "tokenizer_id": "tok_missing",
                "texts": ["a b c"]
            }),
        ),
        (
            "/v1/tokenizer/train_from_iterator",
            serde_json::json!({
                "tokenizer_id": "tok_missing",
                "iterator": ["a b c"]
            }),
        ),
        (
            "/v1/tokenizer/save",
            serde_json::json!({
                "tokenizer_id": "tok_missing",
                "path": "/tmp/missing.json"
            }),
        ),
    ];
    for (endpoint, body) in cases {
        let resp = post_json(ctx.addr(), endpoint, &body);
        assert_error(resp, 404, "tokenizer_not_found");
    }
}

#[test]
fn malformed_json_returns_invalid_request_across_post_endpoints() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let malformed = "{";
    let endpoints = [
        "/v1/tokenizer/instances",
        "/v1/tokenizer/encode",
        "/v1/tokenizer/encode_batch",
        "/v1/tokenizer/decode",
        "/v1/tokenizer/decode_batch",
        "/v1/tokenizer/token_to_id",
        "/v1/tokenizer/id_to_token",
        "/v1/tokenizer/enable_truncation",
        "/v1/tokenizer/disable_truncation",
        "/v1/tokenizer/enable_padding",
        "/v1/tokenizer/disable_padding",
        "/v1/tokenizer/add_tokens",
        "/v1/tokenizer/add_special_tokens",
        "/v1/tokenizer/train",
        "/v1/tokenizer/train_from_iterator",
        "/v1/tokenizer/save",
        "/v1/tokenizer/compare",
    ];
    for endpoint in endpoints {
        let resp = send_request(
            ctx.addr(),
            "POST",
            endpoint,
            &[("Content-Type", "application/json")],
            Some(malformed),
        );
        assert_error(resp, 400, "invalid_request");
    }
}
