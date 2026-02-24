//! Integration tests for project_id support on sessions.

use super::{
    seed_session, seed_session_with_project, session_config,
};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// GET /v1/chat/sessions — project_id filter
// ---------------------------------------------------------------------------

#[test]
fn list_project_id_filters_sessions() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-p1a", "Chat A", "m", "proj_1");
    seed_session_with_project(temp.path(), "sess-p1b", "Chat B", "m", "proj_1");
    seed_session_with_project(temp.path(), "sess-p2a", "Chat C", "m", "proj_2");
    seed_session(temp.path(), "sess-none", "Chat D", "m");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Filter by proj_1
    let resp = get(ctx.addr(), "/v1/chat/sessions?project_id=proj_1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2, "should return 2 sessions for proj_1");
    let ids: Vec<&str> = data.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-p1a"));
    assert!(ids.contains(&"sess-p1b"));
}

#[test]
fn list_project_id_excludes_other_projects() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pa", "Chat", "m", "proj_1");
    seed_session_with_project(temp.path(), "sess-pb", "Chat", "m", "proj_2");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/chat/sessions?project_id=proj_2");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-pb");
}

#[test]
fn list_project_id_nonexistent_returns_empty() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pa", "Chat", "m", "proj_1");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/chat/sessions?project_id=nonexistent");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert!(data.is_empty(), "no sessions should match nonexistent project_id");
}

#[test]
fn list_without_project_id_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-x1", "Chat", "m", "proj_1");
    seed_session_with_project(temp.path(), "sess-x2", "Chat", "m", "proj_2");
    seed_session(temp.path(), "sess-x3", "Chat", "m"); // no project_id

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 3, "all sessions returned without project_id filter");
}

// ---------------------------------------------------------------------------
// project_id in session response
// ---------------------------------------------------------------------------

#[test]
fn list_project_id_shown_in_response() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pshow", "Chat", "m", "my-project");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data[0]["project_id"], "my-project");
}

#[test]
fn get_project_id_shown_in_response() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pget", "Chat", "m", "proj_abc");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-pget");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["project_id"], "proj_abc");
}

#[test]
fn get_session_without_project_id_has_null() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-noproj", "Chat", "m");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-noproj");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert!(json["project_id"].is_null(), "project_id should be null when unset");
}

// ---------------------------------------------------------------------------
// PATCH project_id via metadata
// ---------------------------------------------------------------------------

#[test]
fn patch_metadata_project_id_sets_value() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ppatch", "Chat", "m");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Set project_id via metadata
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-ppatch",
        &json!({"metadata": {"project_id": "proj_new"}}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify via GET (full record includes project_id)
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ppatch");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["project_id"], "proj_new");
}

#[test]
fn patch_metadata_project_id_updates_existing() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pupd", "Chat", "m", "proj_old");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Update project_id
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-pupd",
        &json!({"metadata": {"project_id": "proj_new"}}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify via GET
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-pupd");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["project_id"], "proj_new");
}

#[test]
fn patch_metadata_project_id_null_clears_value() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-pclr", "Chat", "m", "proj_to_clear");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Clear project_id by setting metadata.project_id to null
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-pclr",
        &json!({"metadata": {"project_id": null}}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify via GET
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-pclr");
    assert_eq!(resp.status, 200);
    assert!(
        resp.json()["project_id"].is_null(),
        "project_id should be null after clearing"
    );
}

#[test]
fn patch_top_level_project_id_sets_value() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ptop", "Chat", "m");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Set project_id via top-level field
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-ptop",
        &json!({"project_id": "proj_top"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify via GET
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ptop");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["project_id"], "proj_top");
}

#[test]
fn patch_top_level_project_id_null_clears_value() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "sess-ptopclr", "Chat", "m", "proj_x");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Clear project_id via top-level null
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-ptopclr",
        &json!({"project_id": null}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify via GET
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ptopclr");
    assert_eq!(resp.status, 200);
    assert!(
        resp.json()["project_id"].is_null(),
        "project_id should be null after clearing via top-level null"
    );
}

// ---------------------------------------------------------------------------
// Filter + project_id pagination
// ---------------------------------------------------------------------------

#[test]
fn list_project_id_filter_with_pagination() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session_with_project(
            temp.path(),
            &format!("sess-ppg-{i}"),
            &format!("Chat {i}"),
            "m",
            "paged-proj",
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Add session from another project
    seed_session_with_project(temp.path(), "sess-other", "Other", "m", "other-proj");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Page 1
    let resp = get(
        ctx.addr(),
        "/v1/chat/sessions?project_id=paged-proj&limit=3",
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let page1 = json["data"].as_array().expect("data");
    assert_eq!(page1.len(), 3);
    assert_eq!(json["has_more"], true);
    assert_eq!(json["total"], 5);

    // Page 2: offset=3
    let resp = get(
        ctx.addr(),
        "/v1/chat/sessions?project_id=paged-proj&offset=3&limit=3",
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let page2 = json["data"].as_array().expect("data");
    assert_eq!(page2.len(), 2);
    assert_eq!(json["has_more"], false);
}

// ---------------------------------------------------------------------------
// PATCH project_id persists through list filter
// ---------------------------------------------------------------------------

#[test]
fn patch_project_id_then_filter_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-pfilt", "Chat", "m");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Session initially has no project_id, so filtering should return empty
    let resp = get(ctx.addr(), "/v1/chat/sessions?project_id=proj_assigned");
    assert_eq!(resp.status, 200);
    assert!(resp.json()["data"].as_array().unwrap().is_empty());

    // Assign project_id via PATCH
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-pfilt",
        &json!({"metadata": {"project_id": "proj_assigned"}}),
    );
    assert_eq!(resp.status, 200);

    // Now filtering should find the session
    let resp = get(ctx.addr(), "/v1/chat/sessions?project_id=proj_assigned");
    assert_eq!(resp.status, 200);
    let data = resp.json()["data"].as_array().unwrap().clone();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-pfilt");
}
