//! Cursor-based pagination tests for `GET /v1/files`.

use super::files_config;
use crate::server::common::{get, patch_json, send_request, ServerTestContext};
use tempfile::TempDir;

fn upload_text_file(ctx: &ServerTestContext, filename: &str, mime: &str, payload: &str) -> String {
    let boundary = "----talu-file-list-test";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
        filename = filename,
        mime = mime,
        payload = payload,
    );
    let content_type = format!("multipart/form-data; boundary={}", boundary);
    let headers = [("Content-Type", content_type.as_str())];
    let resp = send_request(ctx.addr(), "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(resp.status, 200, "upload body: {}", resp.body);
    resp.json()["id"].as_str().expect("file id").to_string()
}

#[test]
fn list_cursor_pagination_round_trip() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let mut all_ids: Vec<String> = Vec::new();
    for i in 0..5 {
        let id = upload_text_file(
            &ctx,
            &format!("page-{i}.txt"),
            "text/plain",
            &format!("cursor-round-trip-{i}"),
        );
        all_ids.push(id);
    }

    // Page 1: limit=2
    let resp1 = get(ctx.addr(), "/v1/files?limit=2");
    assert_eq!(resp1.status, 200, "body: {}", resp1.body);
    let json1 = resp1.json();
    let page1 = json1["data"].as_array().expect("page1 data");
    assert_eq!(page1.len(), 2);
    assert_eq!(json1["has_more"], true);
    let cursor = json1["cursor"].as_str().expect("cursor string");

    // Page 2: use cursor
    let resp2 = get(
        ctx.addr(),
        &format!("/v1/files?limit=2&cursor={cursor}"),
    );
    assert_eq!(resp2.status, 200, "body: {}", resp2.body);
    let json2 = resp2.json();
    let page2 = json2["data"].as_array().expect("page2 data");
    assert_eq!(page2.len(), 2);
    assert_eq!(json2["has_more"], true);
    let cursor2 = json2["cursor"].as_str().expect("cursor2 string");

    // Page 3: last page
    let resp3 = get(
        ctx.addr(),
        &format!("/v1/files?limit=2&cursor={cursor2}"),
    );
    assert_eq!(resp3.status, 200, "body: {}", resp3.body);
    let json3 = resp3.json();
    let page3 = json3["data"].as_array().expect("page3 data");
    assert_eq!(page3.len(), 1);
    assert_eq!(json3["has_more"], false);

    // Verify no overlap across pages
    let page1_ids: Vec<&str> = page1.iter().map(|v| v["id"].as_str().unwrap()).collect();
    let page2_ids: Vec<&str> = page2.iter().map(|v| v["id"].as_str().unwrap()).collect();
    let page3_ids: Vec<&str> = page3.iter().map(|v| v["id"].as_str().unwrap()).collect();

    for id in &page2_ids {
        assert!(
            !page1_ids.contains(id),
            "file {id} appears in both page1 and page2"
        );
    }
    for id in &page3_ids {
        assert!(
            !page1_ids.contains(id),
            "file {id} appears in both page1 and page3"
        );
        assert!(
            !page2_ids.contains(id),
            "file {id} appears in both page2 and page3"
        );
    }
}

#[test]
fn list_pagination_covers_all_files() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let mut all_ids: Vec<String> = Vec::new();
    for i in 0..7 {
        let id = upload_text_file(
            &ctx,
            &format!("all-{i}.txt"),
            "text/plain",
            &format!("pagination-all-{i}"),
        );
        all_ids.push(id);
    }

    let mut collected_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..10 {
        let url = match &cursor {
            Some(c) => format!("/v1/files?limit=2&cursor={c}"),
            None => "/v1/files?limit=2".to_string(),
        };
        let resp = get(ctx.addr(), &url);
        assert_eq!(resp.status, 200);
        let json = resp.json();
        let data = json["data"].as_array().expect("data");
        for item in data {
            collected_ids.push(item["id"].as_str().unwrap().to_string());
        }
        if json["has_more"] == false {
            break;
        }
        cursor = Some(json["cursor"].as_str().expect("cursor").to_string());
    }

    assert_eq!(collected_ids.len(), 7, "should collect all 7 files");
    for id in &all_ids {
        assert!(
            collected_ids.contains(id),
            "file {id} not found in paginated results"
        );
    }
}

#[test]
fn list_cursor_null_when_no_more() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    upload_text_file(&ctx, "single.txt", "text/plain", "only-one");

    let resp = get(ctx.addr(), "/v1/files?limit=10");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["has_more"], false);
    assert!(
        json["cursor"].is_null(),
        "cursor should be null when has_more is false"
    );
}

#[test]
fn list_has_more_false_when_exact_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    for i in 0..3 {
        upload_text_file(
            &ctx,
            &format!("exact-{i}.txt"),
            "text/plain",
            &format!("content-exact-{i}"),
        );
    }

    let resp = get(ctx.addr(), "/v1/files?limit=3");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 3);
    assert_eq!(json["has_more"], false);
}

#[test]
fn list_invalid_cursor_ignored() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    upload_text_file(&ctx, "cursor-test.txt", "text/plain", "cursor-ignored-content");

    let resp = get(ctx.addr(), "/v1/files?cursor=not-valid-base64!!!");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
}

#[test]
fn list_marker_filters_files() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    let id1 = upload_text_file(&ctx, "active.txt", "text/plain", "active-content");
    let id2 = upload_text_file(&ctx, "to-archive.txt", "text/plain", "archive-content");

    // Archive id2
    let patch_resp = patch_json(
        ctx.addr(),
        &format!("/v1/files/{id2}"),
        &serde_json::json!({"marker": "archived"}),
    );
    assert_eq!(patch_resp.status, 200, "patch body: {}", patch_resp.body);

    // Default list (marker=active) should only have id1
    let resp = get(ctx.addr(), "/v1/files");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], id1);

    // List with marker=archived should only have id2
    let resp2 = get(ctx.addr(), "/v1/files?marker=archived");
    assert_eq!(resp2.status, 200);
    let json2 = resp2.json();
    let data2 = json2["data"].as_array().expect("data");
    assert_eq!(data2.len(), 1);
    assert_eq!(data2[0]["id"], id2);
}

#[test]
fn list_cursor_with_marker_filter() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    for i in 0..5 {
        upload_text_file(
            &ctx,
            &format!("marker-page-{i}.txt"),
            "text/plain",
            &format!("marker-page-content-{i}"),
        );
    }

    // Page through with limit=2 and marker=active
    let resp1 = get(ctx.addr(), "/v1/files?limit=2&marker=active");
    assert_eq!(resp1.status, 200);
    let json1 = resp1.json();
    let page1 = json1["data"].as_array().expect("page1");
    assert_eq!(page1.len(), 2);
    assert_eq!(json1["has_more"], true);
    let cursor = json1["cursor"].as_str().expect("cursor");

    let resp2 = get(
        ctx.addr(),
        &format!("/v1/files?limit=2&marker=active&cursor={cursor}"),
    );
    assert_eq!(resp2.status, 200);
    let json2 = resp2.json();
    let page2 = json2["data"].as_array().expect("page2");
    assert_eq!(page2.len(), 2);
    assert_eq!(json2["has_more"], true);

    // No overlap
    let p1_ids: Vec<&str> = page1.iter().map(|v| v["id"].as_str().unwrap()).collect();
    let p2_ids: Vec<&str> = page2.iter().map(|v| v["id"].as_str().unwrap()).collect();
    for id in &p2_ids {
        assert!(!p1_ids.contains(id), "overlap between pages");
    }
}

/// Cursor pagination through archived files returns all archived files.
#[test]
fn list_cursor_pagination_archived() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(files_config(temp.path()));

    // Upload 5 files.
    let mut all_ids: Vec<String> = Vec::new();
    for i in 0..5 {
        let id = upload_text_file(
            &ctx,
            &format!("arch-page-{i}.txt"),
            "text/plain",
            &format!("arch-page-content-{i}"),
        );
        all_ids.push(id);
    }

    // Archive all 5.
    for id in &all_ids {
        let resp = patch_json(
            ctx.addr(),
            &format!("/v1/files/{id}"),
            &serde_json::json!({"marker": "archived"}),
        );
        assert_eq!(resp.status, 200, "patch body: {}", resp.body);
    }

    // Page through archived files with limit=2.
    let mut collected_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..10 {
        let url = match &cursor {
            Some(c) => format!("/v1/files?limit=2&marker=archived&cursor={c}"),
            None => "/v1/files?limit=2&marker=archived".to_string(),
        };
        let resp = get(ctx.addr(), &url);
        assert_eq!(resp.status, 200);
        let json = resp.json();
        let data = json["data"].as_array().expect("data");
        for item in data {
            collected_ids.push(item["id"].as_str().unwrap().to_string());
        }
        if json["has_more"] == false {
            break;
        }
        cursor = Some(json["cursor"].as_str().expect("cursor").to_string());
    }

    assert_eq!(collected_ids.len(), 5, "should collect all 5 archived files");
    for id in &all_ids {
        assert!(
            collected_ids.contains(id),
            "archived file {id} not found in paginated results"
        );
    }

    // Active list should be empty.
    let active_resp = get(ctx.addr(), "/v1/files?marker=active");
    assert_eq!(active_resp.status, 200);
    let active_json = active_resp.json();
    let active_data = active_json["data"].as_array().expect("data");
    assert_eq!(active_data.len(), 0, "no active files should remain");
}
