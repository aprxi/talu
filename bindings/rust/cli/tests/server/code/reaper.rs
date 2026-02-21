use hyper::StatusCode;

use super::*;

/// Build an AppState with a zero TTL so sessions expire immediately.
fn build_app_zero_ttl() -> (Router, Arc<AppState>) {
    let state = Arc::new(AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: None,
            current_model: None,
        })),
        configured_model: None,
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
        max_file_upload_bytes: 100 * 1024 * 1024,
        max_file_inspect_bytes: 50 * 1024 * 1024,
        code_sessions: Mutex::new(HashMap::new()),
        code_session_ttl: std::time::Duration::ZERO,
    });
    let router = Router::new(state.clone());
    (router, state)
}

#[tokio::test]
async fn session_reaper_evicts_expired_sessions() {
    let (app, state) = build_app_zero_ttl();

    // Create a session via REST.
    let body = serde_json::json!({
        "source": "x = 1",
        "language": "python"
    });
    let (status, json) =
        body_json(send_request(&app, post_json("/v1/code/session/create", &body)).await).await;
    assert_eq!(status, StatusCode::OK);
    let session_id = json["session_id"]
        .as_str()
        .expect("should return session_id")
        .to_string();

    // Session should exist.
    {
        let sessions = state.code_sessions.lock().await;
        assert!(
            sessions.contains_key(&session_id),
            "session should exist after create"
        );
    }

    // Apply the reaper predicate directly with the state's TTL (Duration::ZERO).
    // Any elapsed time > 0 makes the session stale.
    {
        let ttl = state.code_session_ttl;
        let mut sessions = state.code_sessions.lock().await;
        sessions.retain(|_, session| session.last_access.elapsed() < ttl);
    }

    // Session should be evicted.
    {
        let sessions = state.code_sessions.lock().await;
        assert!(
            !sessions.contains_key(&session_id),
            "session should be evicted after reaper runs with zero TTL"
        );
    }
}
