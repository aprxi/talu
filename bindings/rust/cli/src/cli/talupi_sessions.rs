use anyhow::{bail, Context, Result};
use once_cell::sync::Lazy;
use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};

static HTTP_CLIENT: Lazy<Client> = Lazy::new(Client::new);
const LIST_SCAN_LIMIT: usize = 1000;
const LIST_PAGE_SIZE: usize = 100;

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub responses_json: String,
    pub model: Option<String>,
    pub title: Option<String>,
    pub system_prompt: Option<String>,
    pub metadata: Value,
    pub project_id: Option<String>,
    pub marker: Option<String>,
    pub parent_session_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UpsertSessionRequest {
    pub session_id: String,
    pub responses_json: String,
    pub model: Option<String>,
    pub title: Option<String>,
    pub system_prompt: Option<String>,
    pub metadata: Value,
    pub project_id: Option<String>,
    pub marker: Option<String>,
    pub parent_session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SessionResource {
    #[serde(rename = "id")]
    _id: String,
    #[serde(default)]
    items: Vec<Value>,
    model: Option<String>,
    title: Option<String>,
    system_prompt: Option<String>,
    metadata: Option<Value>,
    project_id: Option<String>,
    marker: Option<String>,
    parent_session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SessionListResponse {
    #[serde(default)]
    data: Vec<SessionListEntry>,
    #[serde(default)]
    has_more: bool,
}

#[derive(Debug, Deserialize)]
struct SessionListEntry {
    id: String,
}

pub fn db_host_from_env() -> Option<String> {
    std::env::var("TALU_DB_HOST")
        .ok()
        .and_then(|raw| normalize_host(&raw))
}

pub fn normalize_host(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        Some(trimmed.to_string())
    } else {
        Some(format!("http://{trimmed}"))
    }
}

pub fn resolve_session_id(db_host: &str, target: &str) -> Result<String> {
    let normalized = target.trim();
    if normalized.is_empty() {
        bail!("session id cannot be empty");
    }

    if get_session(db_host, normalized).is_ok() {
        return Ok(normalized.to_string());
    }

    let mut offset = 0usize;
    let mut matches = Vec::new();

    while offset < LIST_SCAN_LIMIT {
        let url = format!("{db_host}/v1/chat/sessions");
        let response = HTTP_CLIENT
            .get(&url)
            .query(&[
                ("limit", LIST_PAGE_SIZE.to_string()),
                ("offset", offset.to_string()),
            ])
            .send()
            .with_context(|| format!("GET {url}"))?;

        if !response.status().is_success() {
            bail!(
                "failed to list sessions: GET {url} -> {}",
                response.status()
            );
        }

        let list_body = response.text().context("read /v1/chat/sessions response")?;
        let list: SessionListResponse =
            serde_json::from_str(&list_body).context("decode /v1/chat/sessions response")?;

        for entry in &list.data {
            if entry.id.starts_with(normalized) {
                matches.push(entry.id.clone());
            }
        }

        if matches.len() > 1 {
            break;
        }
        if !list.has_more || list.data.is_empty() {
            break;
        }
        offset += LIST_PAGE_SIZE;
    }

    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("session '{normalized}' not found"),
        _ => {
            let display = matches.into_iter().take(5).collect::<Vec<_>>().join(", ");
            bail!("session prefix '{normalized}' is ambiguous; matches: {display}")
        }
    }
}

pub fn get_session(db_host: &str, session_id: &str) -> Result<SessionRecord> {
    let url = session_url(db_host, session_id);
    let response = HTTP_CLIENT
        .get(&url)
        .send()
        .with_context(|| format!("GET {url}"))?;

    match response.status() {
        StatusCode::OK => {
            let session_body = response
                .text()
                .context("read /v1/chat/sessions/{session_id} response")?;
            let session: SessionResource = serde_json::from_str(&session_body)
                .context("decode /v1/chat/sessions/{session_id} response")?;
            let responses_json =
                serde_json::to_string(&session.items).context("encode session items JSON")?;
            let metadata = match session.metadata {
                Some(value) if value.is_object() => value,
                _ => json!({}),
            };
            Ok(SessionRecord {
                responses_json,
                model: session.model,
                title: session.title,
                system_prompt: session.system_prompt,
                metadata,
                project_id: session.project_id,
                marker: session.marker,
                parent_session_id: session.parent_session_id,
            })
        }
        StatusCode::NOT_FOUND => bail!("session '{session_id}' not found"),
        status => bail!("failed to load session: GET {url} -> {status}"),
    }
}

pub fn delete_session(db_host: &str, session_id: &str) -> Result<()> {
    let url = session_url(db_host, session_id);
    let response = HTTP_CLIENT
        .delete(&url)
        .send()
        .with_context(|| format!("DELETE {url}"))?;
    match response.status() {
        StatusCode::OK | StatusCode::NO_CONTENT => Ok(()),
        StatusCode::NOT_FOUND => bail!("session '{session_id}' not found"),
        status => bail!("failed to delete session: DELETE {url} -> {status}"),
    }
}

pub fn upsert_session(db_host: &str, req: &UpsertSessionRequest) -> Result<()> {
    let url = session_url(db_host, &req.session_id);
    let responses_json_value: Value =
        serde_json::from_str(&req.responses_json).context("decode responses_json payload")?;
    let mut payload = json!({
        "session_id": req.session_id,
        "responses_json": responses_json_value,
        "metadata": req.metadata,
        "marker": req.marker.as_deref().unwrap_or("active"),
        "parent_session_id": req.parent_session_id,
        "project_id": req.project_id,
        "model": req.model,
        "title": req.title,
        "system_prompt": req.system_prompt,
    });

    if !payload["metadata"].is_object() {
        payload["metadata"] = json!({});
    }

    let response = HTTP_CLIENT
        .put(&url)
        .header("content-type", "application/json")
        .body(payload.to_string())
        .send()
        .with_context(|| format!("PUT {url}"))?;

    if response.status().is_success() {
        Ok(())
    } else {
        bail!(
            "failed to persist session: PUT {url} -> {}",
            response.status()
        )
    }
}

fn session_url(db_host: &str, session_id: &str) -> String {
    format!("{db_host}/v1/chat/sessions/{session_id}")
}

#[cfg(test)]
mod tests {
    use super::{normalize_host, session_url};

    #[test]
    fn normalize_host_adds_http_scheme() {
        assert_eq!(
            normalize_host("localhost:7258").as_deref(),
            Some("http://localhost:7258")
        );
    }

    #[test]
    fn normalize_host_keeps_explicit_scheme() {
        assert_eq!(
            normalize_host(" https://example.test/ ").as_deref(),
            Some("https://example.test")
        );
    }

    #[test]
    fn session_url_uses_raw_session_id_path_segment() {
        assert_eq!(
            session_url("http://localhost:7258", "sess_1-abc"),
            "http://localhost:7258/v1/chat/sessions/sess_1-abc"
        );
    }
}
