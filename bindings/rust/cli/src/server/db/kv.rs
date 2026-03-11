//! DB kv plane HTTP handlers.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::sync::Arc;

use base64::Engine as _;
use bytes::Bytes;
use futures_util::stream;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

use talu::kv::{
    KvDurability, KvError, KvHandle, KvNamespaceStats, KvPutOptions, KvWatchEventType,
};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;
type SharedKvHandle = Arc<tokio::sync::Mutex<KvHandle>>;
type WatchSubscriberMap = HashMap<String, Arc<AtomicUsize>>;

const KV_BATCH_MAX_ENTRIES: usize = 10_000;
const KV_WATCH_STREAM_CAPACITY: usize = 128;
const KV_WATCH_HEARTBEAT_SECONDS: u64 = 15;
const KV_WATCH_DRAIN_MAX_EVENTS: usize = 128;
const KV_WATCH_POLL_INTERVAL_MS: u64 = 25;

static KV_WATCH_SUBSCRIBERS: Lazy<Mutex<WatchSubscriberMap>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

struct KvWatchQueue {
    state: tokio::sync::Mutex<KvWatchQueueState>,
    notify: tokio::sync::Notify,
}

struct KvWatchQueueState {
    frames: VecDeque<Bytes>,
    closed: bool,
    capacity: usize,
}

impl KvWatchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            state: tokio::sync::Mutex::new(KvWatchQueueState {
                frames: VecDeque::with_capacity(capacity),
                closed: false,
                capacity,
            }),
            notify: tokio::sync::Notify::new(),
        }
    }

    async fn push(&self, frame: Bytes) -> bool {
        let mut state = self.state.lock().await;
        if state.closed {
            return false;
        }
        if state.frames.len() >= state.capacity {
            return false;
        }
        state.frames.push_back(frame);
        drop(state);
        self.notify.notify_one();
        true
    }

    async fn push_gap_and_close(&self, frame: Bytes) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }
        state.frames.clear();
        state.frames.push_back(frame);
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
    }

    async fn close(&self) {
        let mut state = self.state.lock().await;
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
    }

    async fn next_frame(&self) -> Option<Bytes> {
        loop {
            let notified = {
                let mut state = self.state.lock().await;
                if let Some(frame) = state.frames.pop_front() {
                    return Some(frame);
                }
                if state.closed {
                    return None;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }
}

struct WatchSubscriberGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for WatchSubscriberGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

fn watch_subscriber_guard(cache_key: &str) -> WatchSubscriberGuard {
    let counter = {
        let mut counters = KV_WATCH_SUBSCRIBERS.lock().expect("watch subscriber mutex poisoned");
        counters
            .entry(cache_key.to_string())
            .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
            .clone()
    };
    counter.fetch_add(1, Ordering::Relaxed);
    WatchSubscriberGuard { counter }
}

fn watch_subscriber_count(cache_key: &str) -> usize {
    KV_WATCH_SUBSCRIBERS
        .lock()
        .expect("watch subscriber mutex poisoned")
        .get(cache_key)
        .map(|counter| counter.load(Ordering::Relaxed))
        .unwrap_or(0)
}

struct WatchStreamState {
    queue: Arc<KvWatchQueue>,
    _subscriber_guard: WatchSubscriberGuard,
}

fn watch_gap_frame(namespace: &str, gap_type: &str, message: &str) -> Bytes {
    sse_json(
        "gap",
        &serde_json::json!({
            "type": gap_type,
            "message": message,
            "namespace": namespace
        }),
    )
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvEntryResponse {
    pub key: String,
    pub value_len: usize,
    pub value_hex: String,
    pub updated_at_ms: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvListResponse {
    pub namespace: String,
    pub count: usize,
    pub data: Vec<KvEntryResponse>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvPutResponse {
    pub namespace: String,
    pub key: String,
    pub value_len: usize,
    pub durability: String,
    pub ttl_ms: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvDeleteResponse {
    pub namespace: String,
    pub key: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvCompactResponse {
    pub namespace: String,
    pub status: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct KvBatchRequest {
    pub entries: Vec<KvBatchEntryRequest>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct KvBatchEntryRequest {
    pub key: String,
    pub value_base64: String,
    pub durability: Option<String>,
    pub ttl_ms: Option<u64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvBatchResponse {
    pub namespace: String,
    pub requested_count: usize,
    pub applied_count: usize,
    pub coalesced_count: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvNamespaceStatsResponse {
    pub namespace: String,
    pub batched_pending: usize,
    pub batched_max_pending: usize,
    pub batched_max_lag_ms: i64,
    pub batched_next_flush_deadline_ms: i64,
    pub batched_enqueued_writes: u64,
    pub batched_coalesced_writes: u64,
    pub batched_rejected_writes: u64,
    pub batched_flush_count: u64,
    pub batched_flushed_entries: u64,
    pub total_live_entries: usize,
    pub ephemeral_live_entries: usize,
    pub watch_subscribers: usize,
    pub watch_published: u64,
    pub watch_gap_events: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct KvWatchEvent {
    pub seq: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub namespace: String,
    pub key: String,
    pub value_len: usize,
    pub durability: Option<String>,
    pub ttl_ms: Option<u64>,
    pub updated_at_ms: i64,
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/entries",
    tag = "DB::KV",
    responses(
        (status = 200, description = "List KV entries", body = KvListResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (namespace, root) = match parse_namespace_and_root(&state, req.uri().path(), auth.as_ref())
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let entries = match kv.lock().await.list() {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };

    let data = entries
        .into_iter()
        .map(|entry| KvEntryResponse {
            key: entry.key,
            value_len: entry.value.len(),
            value_hex: encode_hex(&entry.value),
            updated_at_ms: entry.updated_at_ms,
        })
        .collect::<Vec<_>>();

    json_response(
        StatusCode::OK,
        &KvListResponse {
            namespace,
            count: data.len(),
            data,
        },
    )
}

#[utoipa::path(
    put,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    request_body(content = String, content_type = "application/octet-stream"),
    responses(
        (status = 200, description = "KV entry upserted", body = KvPutResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key"),
        ("durability" = Option<String>, Query, description = "Write durability: strong|batched|ephemeral"),
        ("ttl_ms" = Option<u64>, Query, description = "TTL in milliseconds (0 means no expiry)")
    )
)]
pub async fn handle_put(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let options = match parse_put_options(query.as_deref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let value = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read request body",
            )
        }
    };

    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    if let Err(err) = kv
        .lock()
        .await
        .put_with_options(&key, value.as_ref(), options)
    {
        return kv_error_response(err);
    }
    json_response(
        StatusCode::OK,
        &KvPutResponse {
            namespace,
            key,
            value_len: value.len(),
            durability: durability_to_str(options.durability).to_string(),
            ttl_ms: options.ttl_ms,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/kv/namespaces/{namespace}/batch",
    tag = "DB::KV",
    request_body = KvBatchRequest,
    responses(
        (status = 200, description = "Batch KV upsert completed", body = KvBatchResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 429, description = "Queue saturated", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_batch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, root) = match parse_namespace_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let payload = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read request body",
            )
        }
    };
    let batch = match serde_json::from_slice::<KvBatchRequest>(&payload) {
        Ok(v) => v,
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Expected JSON body with `entries`",
            )
        }
    };
    if batch.entries.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "entries must be non-empty",
        );
    }
    if batch.entries.len() > KV_BATCH_MAX_ENTRIES {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "entries exceeds batch limit",
        );
    }

    #[derive(Debug)]
    struct ParsedEntry {
        key: String,
        value: Vec<u8>,
        options: KvPutOptions,
    }

    let mut dedup: HashMap<String, ParsedEntry> = HashMap::with_capacity(batch.entries.len());
    let mut order: Vec<String> = Vec::with_capacity(batch.entries.len());
    let mut coalesced_count = 0usize;

    for raw in batch.entries {
        if let Err(resp) = validate_key(&raw.key) {
            return resp;
        }
        let durability = match parse_durability(raw.durability.as_deref()) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let value = match base64::engine::general_purpose::STANDARD.decode(raw.value_base64) {
            Ok(v) => v,
            Err(_) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_argument",
                    "value_base64 must be valid base64",
                )
            }
        };
        let entry = ParsedEntry {
            key: raw.key.clone(),
            value,
            options: KvPutOptions {
                durability,
                ttl_ms: raw.ttl_ms.unwrap_or(0),
            },
        };
        if dedup.insert(raw.key.clone(), entry).is_none() {
            order.push(raw.key);
        } else {
            coalesced_count += 1;
        }
    }

    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let mut applied_count = 0usize;
    for key in order {
        let Some(entry) = dedup.remove(&key) else {
            continue;
        };
        if let Err(err) =
            kv.lock()
                .await
                .put_with_options(&entry.key, entry.value.as_slice(), entry.options)
        {
            return kv_error_response(err);
        }
        applied_count += 1;
    }

    json_response(
        StatusCode::OK,
        &KvBatchResponse {
            namespace,
            requested_count: applied_count + coalesced_count,
            applied_count,
            coalesced_count,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV entry fetched", body = KvEntryResponse),
        (status = 404, description = "Entry not found", body = crate::server::http::ErrorResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key")
    )
)]
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let value = match kv.lock().await.get(&key) {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };
    let Some(value) = value else {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "KV entry does not exist",
        );
    };

    json_response(
        StatusCode::OK,
        &KvEntryResponse {
            key,
            value_len: value.data.len(),
            value_hex: encode_hex(&value.data),
            updated_at_ms: value.updated_at_ms,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/stats",
    tag = "DB::KV",
    responses(
        (status = 200, description = "Namespace runtime stats", body = KvNamespaceStatsResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_stats(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (namespace, root) = match parse_namespace_and_root(&state, req.uri().path(), auth.as_ref())
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let stats = match kv.lock().await.stats() {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };
    let watch_subscribers = watch_subscriber_count(&namespace_cache_key(&root, &namespace));

    json_response(
        StatusCode::OK,
        &build_stats_response(namespace, stats, watch_subscribers),
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/watch",
    tag = "DB::KV",
    responses(
        (status = 200, description = "Namespace watch stream"),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_watch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let accepts_sse = req
        .headers()
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(',')
                .map(|part| part.trim().to_ascii_lowercase())
                .any(|part| part.starts_with("text/event-stream"))
        })
        .unwrap_or(false);
    if !accepts_sse {
        return json_error(
            StatusCode::NOT_ACCEPTABLE,
            "invalid_accept",
            "watch endpoint requires Accept: text/event-stream",
        );
    }

    let (namespace, root) = match parse_namespace_and_root(&state, req.uri().path(), auth.as_ref())
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let cache_key = namespace_cache_key(&root, &namespace);
    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let namespace_for_stream = namespace.clone();
    let queue = Arc::new(KvWatchQueue::new(KV_WATCH_STREAM_CAPACITY));
    let producer_queue = queue.clone();
    let subscriber_guard = watch_subscriber_guard(&cache_key);
    tokio::spawn(async move {
        let mut heartbeat =
            tokio::time::interval(std::time::Duration::from_secs(KV_WATCH_HEARTBEAT_SECONDS));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut poll =
            tokio::time::interval(std::time::Duration::from_millis(KV_WATCH_POLL_INTERVAL_MS));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut after_seq = 0u64;
        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    if !producer_queue.push(Bytes::from_static(b": keepalive\n\n")).await {
                        producer_queue
                            .push_gap_and_close(watch_gap_frame(&namespace_for_stream, "consumer_too_slow", "consumer too slow"))
                            .await;
                        break;
                    }
                }
                _ = poll.tick() => {
                    loop {
                        let batch = match kv.lock().await.watch_drain(after_seq, KV_WATCH_DRAIN_MAX_EVENTS) {
                            Ok(batch) => batch,
                            Err(_) => {
                                producer_queue.close().await;
                                return;
                            }
                        };
                        if batch.lost {
                            producer_queue
                                .push_gap_and_close(watch_gap_frame(&namespace_for_stream, "source_gap", "watch source gap"))
                                .await;
                            break;
                        }
                        if batch.events.is_empty() {
                            break;
                        }
                        for event in batch.events {
                            after_seq = event.seq;
                            if !producer_queue.push(sse_json("event", &watch_event_response(&namespace_for_stream, event))).await {
                                producer_queue
                                    .push_gap_and_close(watch_gap_frame(&namespace_for_stream, "consumer_too_slow", "consumer too slow"))
                                    .await;
                                return;
                            }
                        }
                        if after_seq == 0 {
                            break;
                        }
                    }
                }
            }
        }
    });

    let stream = stream::unfold(
        WatchStreamState {
            queue,
            _subscriber_guard: subscriber_guard,
        },
        |state| async move {
            state.queue
            .next_frame()
            .await
            .map(|chunk| (Ok::<_, Infallible>(Frame::data(chunk)), state))
        },
    );
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

#[utoipa::path(
    delete,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV entry deleted", body = KvDeleteResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key")
    )
)]
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let deleted = match kv.lock().await.delete(&key) {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };

    json_response(
        StatusCode::OK,
        &KvDeleteResponse {
            namespace,
            key,
            deleted,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/kv/namespaces/{namespace}/flush",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV namespace flushed", body = KvCompactResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_flush(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    handle_state_op(state, req.uri().path(), auth.as_ref(), true).await
}

#[utoipa::path(
    post,
    path = "/v1/db/kv/namespaces/{namespace}/compact",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV namespace compacted", body = KvCompactResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_compact(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    handle_state_op(state, req.uri().path(), auth.as_ref(), false).await
}

async fn handle_state_op(
    state: Arc<AppState>,
    path: &str,
    auth: Option<&AuthContext>,
    flush: bool,
) -> Response<BoxBody> {
    let (namespace, root) = match parse_namespace_and_root(&state, path, auth) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let kv = match get_or_open_kv_handle(&state, &root, &namespace).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };
    let result = if flush {
        kv.lock().await.flush()
    } else {
        kv.lock().await.compact()
    };
    if let Err(err) = result {
        return kv_error_response(err);
    }
    json_response(
        StatusCode::OK,
        &KvCompactResponse {
            namespace,
            status: if flush { "flushed" } else { "compacted" }.to_string(),
        },
    )
}

fn parse_namespace_and_root(
    state: &AppState,
    path: &str,
    auth: Option<&AuthContext>,
) -> Result<(String, String), Response<BoxBody>> {
    let Some(stripped) = path.strip_prefix("/v1/db/kv/namespaces/") else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    };
    let namespace_raw = stripped.split('/').next().unwrap_or("");
    if namespace_raw.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    }
    let namespace = percent_encoding::percent_decode_str(namespace_raw)
        .decode_utf8_lossy()
        .into_owned();
    if let Err(resp) = validate_namespace(&namespace) {
        return Err(resp);
    }

    let root = resolve_storage_root(state, auth)?;
    Ok((namespace, root.to_string_lossy().to_string()))
}

fn parse_entry_path_and_root(
    state: &AppState,
    path: &str,
    auth: Option<&AuthContext>,
) -> Result<(String, String, String), Response<BoxBody>> {
    let Some(stripped) = path.strip_prefix("/v1/db/kv/namespaces/") else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    };
    let mut parts = stripped.splitn(3, '/');
    let namespace_raw = parts.next().unwrap_or("");
    let segment = parts.next().unwrap_or("");
    let key_raw = parts.next().unwrap_or("");

    if namespace_raw.is_empty() || segment != "entries" || key_raw.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/db/kv/namespaces/{namespace}/entries/{key}",
        ));
    }

    let namespace = percent_encoding::percent_decode_str(namespace_raw)
        .decode_utf8_lossy()
        .into_owned();
    if let Err(resp) = validate_namespace(&namespace) {
        return Err(resp);
    }
    let key = percent_encoding::percent_decode_str(key_raw)
        .decode_utf8_lossy()
        .into_owned();
    if let Err(resp) = validate_key(&key) {
        return Err(resp);
    }
    let root = resolve_storage_root(state, auth)?;
    Ok((namespace, key, root.to_string_lossy().to_string()))
}

fn parse_put_options(query: Option<&str>) -> Result<KvPutOptions, Response<BoxBody>> {
    let mut options = KvPutOptions::default();
    let Some(query_str) = query else {
        return Ok(options);
    };

    if let Some(raw) = parse_query_param(query_str, "durability") {
        options.durability = parse_durability(Some(raw))?;
    }

    if let Some(raw) = parse_query_param(query_str, "ttl_ms") {
        options.ttl_ms = match raw.parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                return Err(json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_argument",
                    "ttl_ms must be a non-negative integer",
                ))
            }
        };
    }

    Ok(options)
}

fn parse_durability(raw: Option<&str>) -> Result<KvDurability, Response<BoxBody>> {
    let Some(raw) = raw else {
        return Ok(KvDurability::Strong);
    };
    match raw {
        "strong" => Ok(KvDurability::Strong),
        "batched" => Ok(KvDurability::Batched),
        "ephemeral" => Ok(KvDurability::Ephemeral),
        _ => Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "durability must be one of: strong, batched, ephemeral",
        )),
    }
}

fn durability_to_str(d: KvDurability) -> &'static str {
    match d {
        KvDurability::Strong => "strong",
        KvDurability::Batched => "batched",
        KvDurability::Ephemeral => "ephemeral",
    }
}

fn parse_query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find_map(|(k, v)| if k == key { Some(v) } else { None })
}

fn validate_namespace(namespace: &str) -> Result<(), Response<BoxBody>> {
    if namespace.is_empty() || namespace.trim().is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must be non-empty",
        ));
    }
    if namespace.len() > 128 {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must be <= 128 bytes",
        ));
    }
    if namespace == "." || namespace == ".." {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace name is invalid",
        ));
    }
    if namespace.contains('/') || namespace.contains('\\') {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must not contain path separators",
        ));
    }
    if namespace.bytes().any(|b| b == 0 || b.is_ascii_control()) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace contains invalid control characters",
        ));
    }
    Ok(())
}

fn validate_key(key: &str) -> Result<(), Response<BoxBody>> {
    if key.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key must be non-empty",
        ));
    }
    if key.len() > 1024 {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key must be <= 1024 bytes",
        ));
    }
    if key.bytes().any(|b| b == 0 || b.is_ascii_control()) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key contains invalid control characters",
        ));
    }
    Ok(())
}

async fn get_or_open_kv_handle(
    state: &Arc<AppState>,
    root: &str,
    namespace: &str,
) -> Result<SharedKvHandle, Response<BoxBody>> {
    let cache_key = namespace_cache_key(root, namespace);
    {
        let cache = state.kv_handles.lock().await;
        if let Some(existing) = cache.get(&cache_key) {
            return Ok(existing.clone());
        }
    }

    let opened = KvHandle::open(root, namespace).map_err(kv_error_response)?;
    let shared = Arc::new(tokio::sync::Mutex::new(opened));

    let mut cache = state.kv_handles.lock().await;
    if let Some(existing) = cache.get(&cache_key) {
        return Ok(existing.clone());
    }
    cache.insert(cache_key, shared.clone());
    Ok(shared)
}

fn namespace_cache_key(root: &str, namespace: &str) -> String {
    format!("{root}\0{namespace}")
}

fn watch_event_response(namespace: &str, event: talu::kv::KvWatchEvent) -> KvWatchEvent {
    KvWatchEvent {
        seq: event.seq,
        event_type: match event.event_type {
            KvWatchEventType::Put => "put".to_string(),
            KvWatchEventType::Delete => "delete".to_string(),
        },
        namespace: namespace.to_string(),
        key: event.key,
        value_len: event.value_len,
        durability: event
            .durability
            .map(|durability| durability_to_str(durability).to_string()),
        ttl_ms: event.ttl_ms,
        updated_at_ms: event.updated_at_ms,
    }
}

fn build_stats_response(
    namespace: String,
    stats: KvNamespaceStats,
    watch_subscribers: usize,
) -> KvNamespaceStatsResponse {
    KvNamespaceStatsResponse {
        namespace,
        batched_pending: stats.batched_pending,
        batched_max_pending: stats.batched_max_pending,
        batched_max_lag_ms: stats.batched_max_lag_ms,
        batched_next_flush_deadline_ms: stats.batched_next_flush_deadline_ms,
        batched_enqueued_writes: stats.batched_enqueued_writes,
        batched_coalesced_writes: stats.batched_coalesced_writes,
        batched_rejected_writes: stats.batched_rejected_writes,
        batched_flush_count: stats.batched_flush_count,
        batched_flushed_entries: stats.batched_flushed_entries,
        total_live_entries: stats.total_live_entries,
        ephemeral_live_entries: stats.ephemeral_live_entries,
        watch_subscribers,
        watch_published: stats.watch_published,
        watch_gap_events: stats.watch_overwritten,
    }
}

fn resolve_storage_root(
    state: &AppState,
    auth: Option<&AuthContext>,
) -> Result<PathBuf, Response<BoxBody>> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            ))
        }
    };
    let base = match auth {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };
    let root = base.join("kv");
    std::fs::create_dir_all(&root).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        )
    })?;
    Ok(root)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn sse_json<T: Serialize>(event: &str, payload: &T) -> Bytes {
    let data = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn kv_error_response(err: KvError) -> Response<BoxBody> {
    match err {
        KvError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        KvError::Busy(msg) => json_error(StatusCode::CONFLICT, "resource_busy", &msg),
        KvError::ResourceExhausted(msg) => {
            json_error(StatusCode::TOO_MANY_REQUESTS, "resource_exhausted", &msg)
        }
        KvError::Storage(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}

fn json_response<T: serde::Serialize>(status: StatusCode, value: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn kv_watch_queue_emits_gap_and_closes_on_overflow() {
        let queue = KvWatchQueue::new(1);
        assert!(queue.push(Bytes::from_static(b"event: event\n\n")).await);
        assert!(!queue.push(Bytes::from_static(b"event: event2\n\n")).await);

        queue
            .push_gap_and_close(watch_gap_frame("ns", "consumer_too_slow", "consumer too slow"))
            .await;

        let frame = queue.next_frame().await.expect("gap frame");
        let text = String::from_utf8_lossy(&frame);
        assert!(text.contains("event: gap"), "frame:\n{text}");
        assert!(text.contains("\"consumer_too_slow\""), "frame:\n{text}");
        assert!(queue.next_frame().await.is_none());
    }
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}
