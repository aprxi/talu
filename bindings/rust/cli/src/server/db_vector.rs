//! Low-level vector database endpoints.
//!
//! Phase 1-5: collection lifecycle + point mutations + query + stats + compact + changes.

use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};
use talu::VectorStore;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const COLLECTIONS_DIR: &str = "vector";
const COLLECTIONS_FILE: &str = "collections.json";
const COLLECTION_STORES_DIR: &str = "collections";
const COLLECTION_STATE_FILE: &str = "state.json";

const SCHEMA_VERSION: u32 = 1;
const STATE_VERSION: u32 = 1;
const MAX_IDEMPOTENCY_ENTRIES: usize = 2048;
const MAX_DB_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CollectionMetric {
    #[default]
    Dot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NormalizationPolicy {
    #[default]
    None,
    L2,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CollectionIdType {
    #[default]
    U64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub(crate) struct CollectionResponse {
    pub name: String,
    pub dims: u32,
    pub metric: CollectionMetric,
    pub normalization: NormalizationPolicy,
    pub id_type: CollectionIdType,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CollectionListResponse {
    pub data: Vec<CollectionResponse>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateCollectionRequest {
    pub name: String,
    pub dims: u32,
    #[serde(default)]
    pub metric: Option<CollectionMetric>,
    #[serde(default)]
    pub normalization: Option<NormalizationPolicy>,
    #[serde(default)]
    pub id_type: Option<CollectionIdType>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct VectorPointInput {
    #[serde(deserialize_with = "deserialize_u64_from_number_or_string")]
    #[schema(value_type = String, example = "42")]
    pub id: u64,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct AppendPointsRequest {
    #[serde(default)]
    pub dims: Option<u32>,
    pub vectors: Vec<VectorPointInput>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct AppendPointsResponse {
    pub collection: String,
    pub dims: u32,
    pub appended_count: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct UpsertPointsRequest {
    #[serde(default)]
    pub dims: Option<u32>,
    pub vectors: Vec<VectorPointInput>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct UpsertPointsResponse {
    pub collection: String,
    pub dims: u32,
    pub upserted_count: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct IdListRequest {
    #[serde(deserialize_with = "deserialize_id_list")]
    #[schema(value_type = Vec<String>, example = json!(["1", "2"]))]
    pub ids: Vec<u64>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FetchPointsRequest {
    #[serde(deserialize_with = "deserialize_id_list")]
    #[schema(value_type = Vec<String>, example = json!(["1", "2"]))]
    pub ids: Vec<u64>,
    #[serde(default)]
    pub include_values: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FetchedPoint {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FetchPointsResponse {
    pub collection: String,
    pub dims: u32,
    pub data: Vec<FetchedPoint>,
    pub not_found_ids: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DeletePointsResponse {
    pub collection: String,
    pub deleted_count: usize,
    pub not_found_count: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct QueryPointsRequest {
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub queries: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_score: Option<f32>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct QueryMatch {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct QueryResultSet {
    pub query_index: usize,
    pub matches: Vec<QueryMatch>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct QueryPointsResponse {
    pub collection: String,
    pub dims: u32,
    pub results: Vec<QueryResultSet>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CollectionStatsResponse {
    pub collection: String,
    pub dims: u32,
    pub metric: CollectionMetric,
    pub normalization: NormalizationPolicy,
    pub visible_count: usize,
    pub tombstone_count: usize,
    pub segment_count: usize,
    pub total_vector_count: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompactResponse {
    pub collection: String,
    pub dims: u32,
    pub kept_count: usize,
    pub removed_tombstones: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub(crate) struct ChangeEventResponse {
    pub seq: u64,
    pub op: String,
    pub id: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ChangesResponse {
    pub collection: String,
    pub data: Vec<ChangeEventResponse>,
    pub has_more: bool,
    pub next_since: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct CollectionsDisk {
    version: u32,
    collections: Vec<CollectionResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PointState {
    values: Vec<f32>,
    deleted: bool,
    updated_at: i64,
    seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdempotencyRecord {
    request_hash: u64,
    status: u16,
    body: serde_json::Value,
    created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CollectionStateDisk {
    version: u32,
    next_seq: u64,
    points: BTreeMap<String, PointState>,
    changes: Vec<ChangeEventResponse>,
    idempotency: BTreeMap<String, IdempotencyRecord>,
}

#[utoipa::path(
    post,
    path = "/v1/db/collections",
    tag = "DB",
    request_body = CreateCollectionRequest,
    responses(
        (status = 201, body = CollectionResponse),
        (status = 200, body = CollectionResponse)
    )
)]
/// POST /v1/db/collections - Create a collection.
pub async fn handle_create_collection(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };

    let create_req: CreateCollectionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    if let Err(msg) = validate_collection_name(&create_req.name) {
        return json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg);
    }
    if create_req.dims == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "dims must be greater than zero",
        );
    }

    let mut disk = match load_collections(&storage_root) {
        Ok(d) => d,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let metric = create_req.metric.unwrap_or_default();
    let normalization = create_req.normalization.unwrap_or_default();
    let id_type = create_req.id_type.unwrap_or_default();

    if let Some(existing) = disk
        .collections
        .iter()
        .find(|c| c.name == create_req.name)
        .cloned()
    {
        if existing.dims == create_req.dims
            && existing.metric == metric
            && existing.normalization == normalization
            && existing.id_type == id_type
        {
            return json_response(StatusCode::OK, &existing);
        }
        return json_error(
            StatusCode::CONFLICT,
            "collection_conflict",
            "collection exists with different configuration",
        );
    }

    let now = unix_ms_now();
    let created = CollectionResponse {
        name: create_req.name,
        dims: create_req.dims,
        metric,
        normalization,
        id_type,
        created_at: now,
        updated_at: now,
    };
    disk.collections.push(created.clone());
    disk.collections.sort_by(|a, b| a.name.cmp(&b.name));

    if let Err(e) = save_collections(&storage_root, &disk) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    let collection_root = collection_store_root(&storage_root, &created.name);
    if let Err(e) = fs::create_dir_all(&collection_root) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create collection directory: {e}"),
        );
    }

    json_response(StatusCode::CREATED, &created)
}

#[utoipa::path(
    get,
    path = "/v1/db/collections",
    tag = "DB",
    responses((status = 200, body = CollectionListResponse))
)]
/// GET /v1/db/collections - List collections.
pub async fn handle_list_collections(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let mut disk = match load_collections(&storage_root) {
        Ok(d) => d,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };
    disk.collections.sort_by(|a, b| a.name.cmp(&b.name));

    json_response(
        StatusCode::OK,
        &CollectionListResponse {
            data: disk.collections,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/collections/{name}",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    responses((status = 200, body = CollectionResponse))
)]
/// GET /v1/db/collections/{name} - Get a collection by name.
pub async fn handle_get_collection(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(name) = extract_collection_name(req.uri().path()) else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let disk = match load_collections(&storage_root) {
        Ok(d) => d,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    if let Some(collection) = disk.collections.into_iter().find(|c| c.name == name) {
        return json_response(StatusCode::OK, &collection);
    }

    json_error(
        StatusCode::NOT_FOUND,
        "collection_not_found",
        &format!("collection not found: {name}"),
    )
}

#[utoipa::path(
    delete,
    path = "/v1/db/collections/{name}",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    responses((status = 204))
)]
/// DELETE /v1/db/collections/{name} - Delete a collection.
pub async fn handle_delete_collection(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(name) = extract_collection_name(req.uri().path()) else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let mut disk = match load_collections(&storage_root) {
        Ok(d) => d,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let before = disk.collections.len();
    disk.collections.retain(|c| c.name != name);
    if disk.collections.len() == before {
        return json_error(
            StatusCode::NOT_FOUND,
            "collection_not_found",
            &format!("collection not found: {name}"),
        );
    }

    if let Err(e) = save_collections(&storage_root, &disk) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    let root = collection_store_root(&storage_root, name);
    if root.exists() {
        let _ = fs::remove_dir_all(root);
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/points/append",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = AppendPointsRequest,
    responses((status = 200, body = AppendPointsResponse))
)]
/// POST /v1/db/collections/{name}/points/append - Append vectors to a collection.
pub async fn handle_append_points(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let idempotency_key = idempotency_key_from_headers(req.headers());
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/points/append")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };
    let request_hash = hash_bytes(&body);

    let mut state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    if let Some(resp) = try_idempotent_replay(&state, idempotency_key.as_deref(), request_hash) {
        return resp;
    }

    let append_req: AppendPointsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    if let Some(dims) = append_req.dims {
        if dims != collection.dims {
            return json_error(
                StatusCode::BAD_REQUEST,
                "dimension_mismatch",
                "request dims does not match collection dims",
            );
        }
    }

    let mut ids = Vec::with_capacity(append_req.vectors.len());
    let mut values = Vec::<f32>::new();
    values.reserve(append_req.vectors.len() * collection.dims as usize);
    let mut request_ids = HashSet::with_capacity(append_req.vectors.len());

    for point in &append_req.vectors {
        if point.values.len() != collection.dims as usize {
            return json_error(
                StatusCode::BAD_REQUEST,
                "dimension_mismatch",
                "vector length does not match collection dims",
            );
        }
        let key = point.id.to_string();
        if !request_ids.insert(key.clone()) {
            return json_error(
                StatusCode::CONFLICT,
                "id_conflict",
                "append contains duplicate IDs in request payload",
            );
        }
        if state.points.get(&key).is_some_and(|p| !p.deleted) {
            return json_error(
                StatusCode::CONFLICT,
                "id_conflict",
                "append contains IDs that already exist; use upsert",
            );
        }
        ids.push(point.id);
        values.extend(point.values.iter().copied());
    }

    if collection.normalization == NormalizationPolicy::L2 {
        if let Err(e) = normalize_rows_in_place(&mut values, collection.dims as usize) {
            return json_error(StatusCode::BAD_REQUEST, "invalid_argument", &e);
        }
    }

    if let Err(resp) = append_to_store(
        &storage_root,
        &collection_name,
        &ids,
        &values,
        collection.dims,
    ) {
        return resp;
    }

    let now = unix_ms_now();
    for (idx, id) in ids.iter().enumerate() {
        let start = idx * collection.dims as usize;
        let end = start + collection.dims as usize;
        let seq = next_seq(&mut state);
        state.points.insert(
            id.to_string(),
            PointState {
                values: values[start..end].to_vec(),
                deleted: false,
                updated_at: now,
                seq,
            },
        );
        state.changes.push(ChangeEventResponse {
            seq,
            op: "append".to_string(),
            id: id.to_string(),
            timestamp: now,
        });
    }

    let payload = AppendPointsResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        appended_count: ids.len(),
    };

    store_idempotency(
        &mut state,
        idempotency_key.as_deref(),
        request_hash,
        StatusCode::OK,
        serde_json::to_value(&payload).unwrap_or(serde_json::Value::Null),
    );

    if let Err(e) = save_collection_state(&storage_root, &collection_name, &state) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/points/upsert",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = UpsertPointsRequest,
    responses((status = 200, body = UpsertPointsResponse))
)]
/// POST /v1/db/collections/{name}/points/upsert - Upsert vectors in a collection.
pub async fn handle_upsert_points(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let idempotency_key = idempotency_key_from_headers(req.headers());
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/points/upsert")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };
    let request_hash = hash_bytes(&body);

    let mut state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    if let Some(resp) = try_idempotent_replay(&state, idempotency_key.as_deref(), request_hash) {
        return resp;
    }

    let upsert_req: UpsertPointsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    if let Some(dims) = upsert_req.dims {
        if dims != collection.dims {
            return json_error(
                StatusCode::BAD_REQUEST,
                "dimension_mismatch",
                "request dims does not match collection dims",
            );
        }
    }

    let mut ids = Vec::with_capacity(upsert_req.vectors.len());
    let mut values = Vec::<f32>::new();
    values.reserve(upsert_req.vectors.len() * collection.dims as usize);

    for point in &upsert_req.vectors {
        if point.values.len() != collection.dims as usize {
            return json_error(
                StatusCode::BAD_REQUEST,
                "dimension_mismatch",
                "vector length does not match collection dims",
            );
        }
        ids.push(point.id);
        values.extend(point.values.iter().copied());
    }

    if collection.normalization == NormalizationPolicy::L2 {
        if let Err(e) = normalize_rows_in_place(&mut values, collection.dims as usize) {
            return json_error(StatusCode::BAD_REQUEST, "invalid_argument", &e);
        }
    }

    if let Err(resp) = append_to_store(
        &storage_root,
        &collection_name,
        &ids,
        &values,
        collection.dims,
    ) {
        return resp;
    }

    let now = unix_ms_now();
    for (idx, id) in ids.iter().enumerate() {
        let start = idx * collection.dims as usize;
        let end = start + collection.dims as usize;
        let seq = next_seq(&mut state);
        state.points.insert(
            id.to_string(),
            PointState {
                values: values[start..end].to_vec(),
                deleted: false,
                updated_at: now,
                seq,
            },
        );
        state.changes.push(ChangeEventResponse {
            seq,
            op: "upsert".to_string(),
            id: id.to_string(),
            timestamp: now,
        });
    }

    let payload = UpsertPointsResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        upserted_count: ids.len(),
    };

    store_idempotency(
        &mut state,
        idempotency_key.as_deref(),
        request_hash,
        StatusCode::OK,
        serde_json::to_value(&payload).unwrap_or(serde_json::Value::Null),
    );

    if let Err(e) = save_collection_state(&storage_root, &collection_name, &state) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/points/delete",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = IdListRequest,
    responses((status = 200, body = DeletePointsResponse))
)]
/// POST /v1/db/collections/{name}/points/delete - Delete vectors by ID.
pub async fn handle_delete_points(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let idempotency_key = idempotency_key_from_headers(req.headers());
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/points/delete")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };
    let request_hash = hash_bytes(&body);

    let mut state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    if let Some(resp) = try_idempotent_replay(&state, idempotency_key.as_deref(), request_hash) {
        return resp;
    }

    let delete_req: IdListRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let now = unix_ms_now();
    let mut deleted = 0usize;
    let mut not_found = 0usize;
    for id in delete_req.ids {
        let key = id.to_string();
        if state.points.get(&key).is_some_and(|p| !p.deleted) {
            let seq = next_seq(&mut state);
            if let Some(point) = state.points.get_mut(&key) {
                point.deleted = true;
                point.updated_at = now;
                point.seq = seq;
                state.changes.push(ChangeEventResponse {
                    seq,
                    op: "delete".to_string(),
                    id: key,
                    timestamp: now,
                });
                deleted += 1;
            }
        } else {
            not_found += 1;
        }
    }

    let payload = DeletePointsResponse {
        collection: collection_name.to_string(),
        deleted_count: deleted,
        not_found_count: not_found,
    };

    store_idempotency(
        &mut state,
        idempotency_key.as_deref(),
        request_hash,
        StatusCode::OK,
        serde_json::to_value(&payload).unwrap_or(serde_json::Value::Null),
    );

    if let Err(e) = save_collection_state(&storage_root, &collection_name, &state) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    let _ = collection; // keeps collection existence validation explicit
    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/points/fetch",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = FetchPointsRequest,
    responses((status = 200, body = FetchPointsResponse))
)]
/// POST /v1/db/collections/{name}/points/fetch - Fetch vectors by ID.
pub async fn handle_fetch_points(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/points/fetch")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };

    let fetch_req: FetchPointsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let include_values = fetch_req.include_values.unwrap_or(true);
    let state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let mut data = Vec::new();
    let mut not_found = Vec::new();

    for id in fetch_req.ids {
        let key = id.to_string();
        match state.points.get(&key) {
            Some(point) if !point.deleted => data.push(FetchedPoint {
                id: key,
                values: if include_values {
                    Some(point.values.clone())
                } else {
                    None
                },
            }),
            _ => not_found.push(key),
        }
    }

    json_response(
        StatusCode::OK,
        &FetchPointsResponse {
            collection: collection_name.to_string(),
            dims: collection.dims,
            data,
            not_found_ids: not_found,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/points/query",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = QueryPointsRequest,
    responses((status = 200, body = QueryPointsResponse))
)]
/// POST /v1/db/collections/{name}/points/query - Query vectors from a collection.
pub async fn handle_query_points(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/points/query")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };

    let mut query_req: QueryPointsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let top_k = query_req.top_k.unwrap_or(10);
    if top_k == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "top_k must be greater than zero",
        );
    }

    let mut queries = Vec::<Vec<f32>>::new();
    match (query_req.vector.take(), query_req.queries.take()) {
        (Some(v), None) => queries.push(v),
        (None, Some(batch)) => queries = batch,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "provide exactly one of: vector, queries",
            )
        }
    }

    if queries.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "queries must be non-empty",
        );
    }

    for q in &queries {
        if q.len() != collection.dims as usize {
            return json_error(
                StatusCode::BAD_REQUEST,
                "dimension_mismatch",
                "query length does not match collection dims",
            );
        }
    }

    if collection.normalization == NormalizationPolicy::L2 {
        for q in &mut queries {
            if let Err(e) = normalize_vector_in_place(q) {
                return json_error(StatusCode::BAD_REQUEST, "invalid_argument", &e);
            }
        }
    }

    let min_score = query_req.min_score;
    let state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let active: Vec<(String, &PointState)> = state
        .points
        .iter()
        .filter(|(_, p)| !p.deleted)
        .map(|(id, p)| (id.clone(), p))
        .collect();

    let mut results = Vec::with_capacity(queries.len());
    for (qi, query) in queries.iter().enumerate() {
        let mut matches = Vec::new();
        for (id, point) in &active {
            let score = dot_product(query, &point.values);
            if min_score.is_some_and(|min| score < min) {
                continue;
            }
            matches.push(QueryMatch {
                id: id.clone(),
                score,
            });
        }

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        if matches.len() > top_k as usize {
            matches.truncate(top_k as usize);
        }

        results.push(QueryResultSet {
            query_index: qi,
            matches,
        });
    }

    json_response(
        StatusCode::OK,
        &QueryPointsResponse {
            collection: collection_name.to_string(),
            dims: collection.dims,
            results,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/collections/{name}/stats",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    responses((status = 200, body = CollectionStatsResponse))
)]
/// GET /v1/db/collections/{name}/stats - Get collection stats.
pub async fn handle_collection_stats(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/stats").map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let mut visible = 0usize;
    let mut tombstones = 0usize;
    for point in state.points.values() {
        if point.deleted {
            tombstones += 1;
        } else {
            visible += 1;
        }
    }

    let segments = count_vector_segments(&collection_store_root(&storage_root, &collection_name));

    json_response(
        StatusCode::OK,
        &CollectionStatsResponse {
            collection: collection_name.to_string(),
            dims: collection.dims,
            metric: collection.metric,
            normalization: collection.normalization,
            visible_count: visible,
            tombstone_count: tombstones,
            segment_count: segments,
            total_vector_count: visible + tombstones,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/compact",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    responses((status = 200, body = CompactResponse))
)]
/// POST /v1/db/collections/{name}/compact - Compact collection state and vector store.
pub async fn handle_compact_collection(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let idempotency_key = idempotency_key_from_headers(req.headers());
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/compact").map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    let collection = match require_collection(&storage_root, &collection_name) {
        Ok(c) => c,
        Err(resp) => return resp,
    };

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };
    let request_hash = hash_bytes(&body);

    let mut state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    if let Some(resp) = try_idempotent_replay(&state, idempotency_key.as_deref(), request_hash) {
        return resp;
    }

    let collection_root = collection_store_root(&storage_root, &collection_name);
    let vector_ns = collection_root.join("vector");
    if vector_ns.exists() {
        if let Err(e) = fs::remove_dir_all(&vector_ns) {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &format!("failed to clear vector namespace: {e}"),
            );
        }
    }

    if let Err(e) = fs::create_dir_all(&collection_root) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create collection root: {e}"),
        );
    }

    let active: Vec<(u64, Vec<f32>)> = state
        .points
        .iter()
        .filter_map(|(id, p)| {
            if p.deleted {
                None
            } else {
                id.parse::<u64>()
                    .ok()
                    .map(|parsed| (parsed, p.values.clone()))
            }
        })
        .collect();

    if !active.is_empty() {
        let mut ids = Vec::with_capacity(active.len());
        let mut values = Vec::with_capacity(active.len() * collection.dims as usize);
        for (id, vector) in &active {
            ids.push(*id);
            values.extend(vector);
        }
        if let Err(resp) = append_to_store(
            &storage_root,
            &collection_name,
            &ids,
            &values,
            collection.dims,
        ) {
            return resp;
        }
    }

    let before_tombstones = state.points.values().filter(|p| p.deleted).count();
    state.points.retain(|_, p| !p.deleted);

    let now = unix_ms_now();
    let seq = next_seq(&mut state);
    state.changes.push(ChangeEventResponse {
        seq,
        op: "compact".to_string(),
        id: "__collection__".to_string(),
        timestamp: now,
    });

    let payload = CompactResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        kept_count: state.points.len(),
        removed_tombstones: before_tombstones,
    };

    store_idempotency(
        &mut state,
        idempotency_key.as_deref(),
        request_hash,
        StatusCode::OK,
        serde_json::to_value(&payload).unwrap_or(serde_json::Value::Null),
    );

    if let Err(e) = save_collection_state(&storage_root, &collection_name, &state) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }

    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(
    get,
    path = "/v1/db/collections/{name}/changes",
    tag = "DB",
    params(
        ("name" = String, Path, description = "Collection name"),
        ("since" = Option<u64>, Query, description = "Return changes with seq > since"),
        ("limit" = Option<usize>, Query, description = "Maximum changes to return (default 100)")
    ),
    responses((status = 200, body = ChangesResponse))
)]
/// GET /v1/db/collections/{name}/changes - Read collection change feed.
pub async fn handle_collection_changes(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/changes").map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    if let Err(resp) = require_collection(&storage_root, &collection_name).map(|_| ()) {
        return resp;
    }

    let query = req.uri().query().unwrap_or("");
    let since = parse_query_param(query, "since")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let limit = parse_query_param(query, "limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100)
        .clamp(1, 1000);

    let state = match load_collection_state(&storage_root, &collection_name) {
        Ok(s) => s,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let filtered: Vec<ChangeEventResponse> = state
        .changes
        .iter()
        .filter(|c| c.seq > since)
        .take(limit + 1)
        .cloned()
        .collect();

    let has_more = filtered.len() > limit;
    let data = if has_more {
        filtered[..limit].to_vec()
    } else {
        filtered.clone()
    };

    let next_since = data.last().map(|c| c.seq).unwrap_or(since);

    json_response(
        StatusCode::OK,
        &ChangesResponse {
            collection: collection_name.to_string(),
            data,
            has_more,
            next_since,
        },
    )
}

fn resolve_storage_root(
    state: &AppState,
    auth: &Option<AuthContext>,
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

    let root = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    if let Err(e) = fs::create_dir_all(&root) {
        return Err(json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        ));
    }

    Ok(root)
}

fn collections_file_path(storage_root: &Path) -> PathBuf {
    storage_root.join(COLLECTIONS_DIR).join(COLLECTIONS_FILE)
}

fn collection_store_root(storage_root: &Path, collection_name: &str) -> PathBuf {
    storage_root
        .join(COLLECTIONS_DIR)
        .join(COLLECTION_STORES_DIR)
        .join(collection_name)
}

fn collection_state_path(storage_root: &Path, collection_name: &str) -> PathBuf {
    collection_store_root(storage_root, collection_name).join(COLLECTION_STATE_FILE)
}

fn load_collections(storage_root: &Path) -> Result<CollectionsDisk, String> {
    let path = collections_file_path(storage_root);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CollectionsDisk {
                version: SCHEMA_VERSION,
                collections: Vec::new(),
            })
        }
        Err(e) => return Err(format!("failed to read collections metadata: {e}")),
    };

    let mut disk: CollectionsDisk = serde_json::from_slice(&bytes)
        .map_err(|e| format!("failed to parse collections metadata: {e}"))?;
    if disk.version == 0 {
        disk.version = SCHEMA_VERSION;
    }
    Ok(disk)
}

fn save_collections(storage_root: &Path, disk: &CollectionsDisk) -> Result<(), String> {
    let dir = storage_root.join(COLLECTIONS_DIR);
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create collections dir: {e}"))?;

    let mut to_write = CollectionsDisk {
        version: SCHEMA_VERSION,
        collections: disk.collections.clone(),
    };
    to_write.collections.sort_by(|a, b| a.name.cmp(&b.name));

    let data = serde_json::to_vec_pretty(&to_write)
        .map_err(|e| format!("failed to encode collections metadata: {e}"))?;

    let path = collections_file_path(storage_root);
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, data).map_err(|e| format!("failed to write collections temp file: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("failed to replace collections metadata: {e}"))?;
    Ok(())
}

fn load_collection_state(
    storage_root: &Path,
    collection_name: &str,
) -> Result<CollectionStateDisk, String> {
    let path = collection_state_path(storage_root, collection_name);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CollectionStateDisk {
                version: STATE_VERSION,
                next_seq: 1,
                points: BTreeMap::new(),
                changes: Vec::new(),
                idempotency: BTreeMap::new(),
            })
        }
        Err(e) => return Err(format!("failed to read collection state: {e}")),
    };

    let mut disk: CollectionStateDisk = serde_json::from_slice(&bytes)
        .map_err(|e| format!("failed to parse collection state: {e}"))?;
    if disk.version == 0 {
        disk.version = STATE_VERSION;
    }
    if disk.next_seq == 0 {
        disk.next_seq = 1;
    }
    Ok(disk)
}

fn save_collection_state(
    storage_root: &Path,
    collection_name: &str,
    state: &CollectionStateDisk,
) -> Result<(), String> {
    let collection_root = collection_store_root(storage_root, collection_name);
    fs::create_dir_all(&collection_root)
        .map_err(|e| format!("failed to create collection root: {e}"))?;

    let mut to_write = state.clone();
    to_write.version = STATE_VERSION;

    let data = serde_json::to_vec_pretty(&to_write)
        .map_err(|e| format!("failed to encode collection state: {e}"))?;

    let path = collection_state_path(storage_root, collection_name);
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, data)
        .map_err(|e| format!("failed to write collection state temp file: {e}"))?;
    fs::rename(&tmp, &path).map_err(|e| format!("failed to replace collection state: {e}"))?;
    Ok(())
}

fn require_collection(
    storage_root: &Path,
    name: &str,
) -> Result<CollectionResponse, Response<BoxBody>> {
    match get_collection_by_name(storage_root, name) {
        Ok(Some(c)) => Ok(c),
        Ok(None) => Err(json_error(
            StatusCode::NOT_FOUND,
            "collection_not_found",
            &format!("collection not found: {name}"),
        )),
        Err(e) => Err(json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &e,
        )),
    }
}

fn get_collection_by_name(
    storage_root: &Path,
    name: &str,
) -> Result<Option<CollectionResponse>, String> {
    let disk = load_collections(storage_root)?;
    Ok(disk.collections.into_iter().find(|c| c.name == name))
}

fn append_to_store(
    storage_root: &Path,
    collection_name: &str,
    ids: &[u64],
    values: &[f32],
    dims: u32,
) -> Result<(), Response<BoxBody>> {
    if ids.is_empty() {
        return Ok(());
    }
    let root = collection_store_root(storage_root, collection_name);
    if let Err(e) = fs::create_dir_all(&root) {
        return Err(json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to prepare collection storage: {e}"),
        ));
    }

    let root_str = root.display().to_string();
    let store = VectorStore::open(&root_str).map_err(vector_error_response)?;
    store
        .append(ids, values, dims)
        .map_err(vector_error_response)?;
    Ok(())
}

fn count_vector_segments(collection_root: &Path) -> usize {
    let vector_ns = collection_root.join("vector");
    let Ok(entries) = fs::read_dir(vector_ns) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".talu"))
        })
        .count()
}

fn next_seq(state: &mut CollectionStateDisk) -> u64 {
    let current = state.next_seq;
    state.next_seq = state.next_seq.saturating_add(1).max(1);
    current
}

fn idempotency_key_from_headers(headers: &hyper::HeaderMap) -> Option<String> {
    headers
        .get("idempotency-key")
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned)
}

fn try_idempotent_replay(
    state: &CollectionStateDisk,
    key: Option<&str>,
    request_hash: u64,
) -> Option<Response<BoxBody>> {
    let key = key?;
    let existing = state.idempotency.get(key)?;
    if existing.request_hash != request_hash {
        return Some(json_error(
            StatusCode::CONFLICT,
            "idempotency_conflict",
            "idempotency key was used with a different request payload",
        ));
    }

    let status = StatusCode::from_u16(existing.status).unwrap_or(StatusCode::OK);
    Some(json_response(status, &existing.body))
}

fn store_idempotency(
    state: &mut CollectionStateDisk,
    key: Option<&str>,
    request_hash: u64,
    status: StatusCode,
    body: serde_json::Value,
) {
    let Some(key) = key else {
        return;
    };

    state.idempotency.insert(
        key.to_string(),
        IdempotencyRecord {
            request_hash,
            status: status.as_u16(),
            body,
            created_at: unix_ms_now(),
        },
    );

    if state.idempotency.len() > MAX_IDEMPOTENCY_ENTRIES {
        let mut items: Vec<(String, i64)> = state
            .idempotency
            .iter()
            .map(|(k, v)| (k.clone(), v.created_at))
            .collect();
        items.sort_by(|a, b| a.1.cmp(&b.1));

        let remove_count = state.idempotency.len() - MAX_IDEMPOTENCY_ENTRIES;
        for (key_to_remove, _) in items.into_iter().take(remove_count) {
            state.idempotency.remove(&key_to_remove);
        }
    }
}

fn validate_collection_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("name must be non-empty".to_string());
    }
    if name.contains('/') || name.contains('\\') {
        return Err("name must not contain path separators".to_string());
    }
    Ok(())
}

fn extract_collection_name(path: &str) -> Option<&str> {
    let stripped = path
        .strip_prefix("/v1/db/collections/")
        .or_else(|| path.strip_prefix("/db/collections/"))?;
    if stripped.is_empty() || stripped.contains('/') {
        return None;
    }
    Some(stripped)
}

fn extract_collection_name_with_suffix<'a>(path: &'a str, suffix: &str) -> Option<&'a str> {
    let stripped = path
        .strip_prefix("/v1/db/collections/")
        .or_else(|| path.strip_prefix("/db/collections/"))?;
    let name = stripped.strip_suffix(suffix)?;
    if name.is_empty() || name.contains('/') {
        return None;
    }
    Some(name)
}

fn normalize_vector_in_place(v: &mut [f32]) -> Result<(), String> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return Err("zero vector cannot be L2-normalized".to_string());
    }
    for x in v {
        *x /= norm;
    }
    Ok(())
}

fn normalize_rows_in_place(flat: &mut [f32], dims: usize) -> Result<(), String> {
    if dims == 0 {
        return Err("dims must be greater than zero".to_string());
    }
    if flat.len() % dims != 0 {
        return Err("flat vector buffer length is not a multiple of dims".to_string());
    }
    for row in flat.chunks_mut(dims) {
        normalize_vector_in_place(row)?;
    }
    Ok(())
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn hash_bytes(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

fn deserialize_u64_from_number_or_string<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = serde_json::Value::deserialize(deserializer)?;
    parse_u64_value(raw).map_err(D::Error::custom)
}

fn deserialize_id_list<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = serde_json::Value::deserialize(deserializer)?;
    let arr = raw
        .as_array()
        .ok_or_else(|| D::Error::custom("ids must be an array"))?;

    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(parse_u64_value(item.clone()).map_err(D::Error::custom)?);
    }
    Ok(out)
}

fn parse_u64_value(value: serde_json::Value) -> Result<u64, String> {
    match value {
        serde_json::Value::Number(n) => n
            .as_u64()
            .ok_or_else(|| "id number must be an unsigned 64-bit integer".to_string()),
        serde_json::Value::String(s) => s
            .parse::<u64>()
            .map_err(|_| "id string must parse as unsigned 64-bit integer".to_string()),
        _ => Err("id must be number or string".to_string()),
    }
}

fn parse_query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find_map(|(k, v)| if k == key { Some(v) } else { None })
}

fn unix_ms_now() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

#[derive(Debug)]
enum BodyReadError {
    TooLarge { max_bytes: usize },
    Read(String),
}

fn body_read_error_response(err: BodyReadError) -> Response<BoxBody> {
    match err {
        BodyReadError::TooLarge { max_bytes } => json_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "resource_exhausted",
            &format!("request body exceeds maximum size ({max_bytes} bytes)"),
        ),
        BodyReadError::Read(msg) => json_error(StatusCode::BAD_REQUEST, "invalid_body", &msg),
    }
}

async fn read_body(req: Request<Incoming>, max_bytes: usize) -> Result<Vec<u8>, BodyReadError> {
    let mut body = req.into_body();
    let mut out = Vec::new();

    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|e| BodyReadError::Read(format!("Failed to read body: {e}")))?;
        if let Some(data) = frame.data_ref() {
            let new_len = out.len().saturating_add(data.len());
            if new_len > max_bytes {
                return Err(BodyReadError::TooLarge { max_bytes });
            }
            out.extend_from_slice(data);
        }
    }

    Ok(out)
}

fn json_response<T: Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
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

fn vector_error_response(err: talu::VectorError) -> Response<BoxBody> {
    match err {
        talu::VectorError::InvalidArgument(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("dim") {
                json_error(StatusCode::BAD_REQUEST, "dimension_mismatch", &msg)
            } else {
                json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
            }
        }
        talu::VectorError::StoreError(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("invalidcolumndata") || lower.contains("dim") {
                json_error(StatusCode::BAD_REQUEST, "dimension_mismatch", &msg)
            } else {
                json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
            }
        }
    }
}
