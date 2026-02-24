use std::io::Read;
use std::net::TcpStream;
use std::time::Duration;

use crate::server::common::{get, send_request, ServerConfig, ServerTestContext, TenantSpec};

#[test]
fn replay_returns_event_envelopes() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let _ = get(ctx.addr(), "/health");

    let resp = get(ctx.addr(), "/v1/events?verbosity=1&limit=64");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let events = json["events"].as_array().expect("events array");
    assert!(!events.is_empty(), "expected at least one replay event");

    let first = &events[0];
    for key in [
        "id",
        "cursor",
        "ts_ms",
        "verbosity_min",
        "level",
        "domain",
        "topic",
        "event_class",
        "message",
        "tenant_id",
        "correlation",
        "data",
    ] {
        assert!(first.get(key).is_some(), "missing key `{key}` in envelope");
    }
}

#[test]
fn replay_rejects_malformed_cursor() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/v1/events?cursor=bad-cursor");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["code"].as_str(), Some("invalid_request"));
}

#[test]
fn stream_serves_event_sse_frames() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let _ = get(ctx.addr(), "/health");

    let mut stream = TcpStream::connect(ctx.addr()).expect("connect stream");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let request = format!(
        "GET /v1/events/stream?verbosity=1 HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        ctx.addr()
    );
    use std::io::Write;
    stream
        .write_all(request.as_bytes())
        .expect("write stream request");

    let mut buf = [0u8; 8192];
    let mut out = Vec::new();
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                if out.windows(8).any(|w| w == b"event: e")
                    && out.windows(6).any(|w| w == b"data: ")
                {
                    break;
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => break,
            Err(err) => panic!("failed reading stream response: {err}"),
        }
    }

    let text = String::from_utf8_lossy(&out);
    assert!(
        text.contains("text/event-stream"),
        "expected SSE headers, got:\n{text}"
    );
    assert!(
        text.contains("event: event"),
        "expected event frame, got:\n{text}"
    );
    assert!(text.contains("data: "), "expected data line, got:\n{text}");
}

#[test]
fn topics_endpoint_lists_supported_filters() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/v1/events/topics");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let fields = json["filter_fields"]
        .as_array()
        .expect("filter_fields array")
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>();
    for expected in [
        "verbosity",
        "domains",
        "topics",
        "event_class",
        "response_id",
        "session_id",
        "cursor",
        "limit",
    ] {
        assert!(
            fields.contains(&expected),
            "missing filter field {expected}"
        );
    }
}

#[test]
fn tenant_replay_isolation_excludes_other_tenants() {
    let mut cfg = ServerConfig::new();
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];
    let ctx = ServerTestContext::new(cfg);

    let acme = send_request(
        ctx.addr(),
        "GET",
        "/v1/events?verbosity=3&limit=128",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
        ],
        None,
    );
    assert_eq!(acme.status, 200, "body: {}", acme.body);
    let acme_events = acme.json()["events"]
        .as_array()
        .expect("acme events")
        .clone();
    assert!(
        acme_events
            .iter()
            .any(|e| e["tenant_id"].as_str() == Some("acme")),
        "expected at least one tenant-scoped acme event",
    );
    assert!(
        acme_events
            .iter()
            .all(|e| e["tenant_id"].as_str() != Some("globex")),
        "acme response leaked globex event",
    );

    let globex = send_request(
        ctx.addr(),
        "GET",
        "/v1/events?verbosity=3&limit=128",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
        ],
        None,
    );
    assert_eq!(globex.status, 200, "body: {}", globex.body);
    let globex_events = globex.json()["events"]
        .as_array()
        .expect("globex events")
        .clone();
    assert!(
        globex_events
            .iter()
            .any(|e| e["tenant_id"].as_str() == Some("globex")),
        "expected at least one tenant-scoped globex event",
    );
    assert!(
        globex_events
            .iter()
            .all(|e| e["tenant_id"].as_str() != Some("acme")),
        "globex response leaked acme event",
    );
}
