use std::path::PathBuf;

use base64::Engine;
use tempfile::TempDir;

use crate::server::common::*;

fn collab_config(bucket: PathBuf) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.bucket = Some(bucket);
    config
}

#[test]
fn collab_docs_and_openapi_are_served() {
    let ctx = ServerTestContext::new(ServerConfig::new());

    let docs = get(ctx.addr(), "/docs");
    assert_eq!(docs.status, 200, "body: {}", docs.body);
    assert!(docs.body.contains("/docs/collab/resources"));

    let resp = get(ctx.addr(), "/openapi/collab/resources.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(resp
        .body
        .contains("/v1/collab/resources/{kind}/{id}/sessions"));
}

#[test]
fn collab_resource_roundtrip_works() {
    let bucket = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(collab_config(bucket.path().to_path_buf()));

    let open = post_json(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/sessions",
        &serde_json::json!({
            "participant_id": "human:1",
            "participant_kind": "human",
            "role": "editor"
        }),
    );
    assert_eq!(open.status, 200, "body: {}", open.body);
    let open_json = open.json();
    assert_eq!(open_json["participant_id"], "human:1");
    assert_eq!(open_json["status"], "joined");

    let submit = post_json(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/ops",
        &serde_json::json!({
            "actor_id": "human:1",
            "actor_seq": 1,
            "op_id": "op-1",
            "payload_base64": base64::engine::general_purpose::STANDARD.encode(br#"{"type":"insert"}"#),
            "snapshot_base64": base64::engine::general_purpose::STANDARD.encode(b"hello"),
            "issued_at_ms": 1,
            "durability": "strong"
        }),
    );
    assert_eq!(submit.status, 200, "body: {}", submit.body);
    assert!(
        submit.body.contains("ops/human:1:1:op-1"),
        "body: {}",
        submit.body
    );

    let summary = get(ctx.addr(), "/v1/collab/resources/text_document/doc-1");
    assert_eq!(summary.status, 200, "body: {}", summary.body);
    assert!(summary.body.contains("\"resource_kind\":\"text_document\""));

    let snapshot = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/snapshot",
    );
    assert_eq!(snapshot.status, 200, "body: {}", snapshot.body);
    let snapshot_json = snapshot.json();
    assert_eq!(
        snapshot_json["snapshot_base64"],
        base64::engine::general_purpose::STANDARD.encode(b"hello")
    );

    let history = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/history",
    );
    assert_eq!(history.status, 200, "body: {}", history.body);
    assert!(history.body.contains("\"op_id\":\"op-1\""));

    let presence_put = post_json(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/presence/human%3A1",
        &serde_json::json!({
            "presence": { "cursor": 42 }
        }),
    );
    assert_eq!(presence_put.status, 200, "body: {}", presence_put.body);

    let presence_get = get(
        ctx.addr(),
        "/v1/collab/resources/text_document/doc-1/presence/human%3A1",
    );
    assert_eq!(presence_get.status, 200, "body: {}", presence_get.body);
    let presence_json = presence_get.json();
    assert_eq!(presence_json["presence"]["cursor"], 42);
}
