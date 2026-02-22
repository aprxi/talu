//! Low-level vector database endpoints.
//!

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use once_cell::sync::Lazy;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};
use talu::VectorStore;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const COLLECTIONS_DIR: &str = "vector";
const COLLECTIONS_FILE: &str = "collections.json";
const COLLECTION_STORES_DIR: &str = "collections";

const SCHEMA_VERSION: u32 = 1;
const MAX_DB_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_INDEX_BUILD_MAX_SEGMENTS: usize = 32;

static COLLECTIONS_CACHE: Lazy<RwLock<HashMap<PathBuf, CollectionsDisk>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

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
    #[serde(default)]
    pub approximate: Option<bool>,
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
    pub manifest_generation: u64,
    pub index_ready_segments: usize,
    pub index_pending_segments: usize,
    pub index_failed_segments: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompactResponse {
    pub collection: String,
    pub dims: u32,
    pub kept_count: usize,
    pub removed_tombstones: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CompactCollectionRequest {
    #[serde(default)]
    pub expected_generation: Option<u64>,
    #[serde(default)]
    pub ttl_max_age_ms: Option<i64>,
    #[serde(default)]
    pub now_ms: Option<i64>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct BuildIndexesRequest {
    pub expected_generation: u64,
    #[serde(default)]
    pub max_segments: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BuildIndexesResponse {
    pub collection: String,
    pub built_segments: usize,
    pub failed_segments: usize,
    pub pending_segments: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CollectionsDisk {
    version: u32,
    collections: Vec<CollectionResponse>,
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

    let _collections_lock = match acquire_file_lock(&collections_lock_path(&storage_root)) {
        Ok(lock) => lock,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let mut disk = match load_collections_cached(&storage_root) {
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
    if let Err(e) = store_collections_cache(&storage_root, &disk) {
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

    let mut disk = match load_collections_cached(&storage_root) {
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

    let disk = match load_collections_cached(&storage_root) {
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

    let _collections_lock = match acquire_file_lock(&collections_lock_path(&storage_root)) {
        Ok(lock) => lock,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
    };

    let mut disk = match load_collections_cached(&storage_root) {
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
    if let Err(e) = store_collections_cache(&storage_root, &disk) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e);
    }
    clear_collection_store_cache(&storage_root, name);

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
    let (idempotency_key_hash, idempotency_request_hash) =
        idempotency_hashes(idempotency_key.as_deref(), request_hash);

    let _collection_lock =
        match acquire_file_lock(&collection_lock_path(&storage_root, &collection_name)) {
            Ok(lock) => lock,
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
        };

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
        if !request_ids.insert(point.id) {
            return json_error(
                StatusCode::CONFLICT,
                "id_conflict",
                "append contains duplicate IDs in request payload",
            );
        }
        ids.push(point.id);
        values.extend(point.values.iter().copied());
    }

    let append_result = with_collection_store(&storage_root, &collection_name, |store| {
        store
            .append_idempotent_with_options(
                &ids,
                &values,
                collection.dims,
                collection.normalization == NormalizationPolicy::L2,
                true,
                idempotency_key_hash,
                idempotency_request_hash,
            )
            .map_err(vector_error_response)
    });
    if let Err(resp) = append_result {
        return resp;
    }

    let payload = AppendPointsResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        appended_count: ids.len(),
    };

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
    let (idempotency_key_hash, idempotency_request_hash) =
        idempotency_hashes(idempotency_key.as_deref(), request_hash);

    let _collection_lock =
        match acquire_file_lock(&collection_lock_path(&storage_root, &collection_name)) {
            Ok(lock) => lock,
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
        };

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

    let upsert_result = with_collection_store(&storage_root, &collection_name, |store| {
        store
            .upsert_idempotent_with_options(
                &ids,
                &values,
                collection.dims,
                collection.normalization == NormalizationPolicy::L2,
                idempotency_key_hash,
                idempotency_request_hash,
            )
            .map_err(vector_error_response)
    });
    if let Err(resp) = upsert_result {
        return resp;
    }

    let payload = UpsertPointsResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        upserted_count: ids.len(),
    };

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
    let (idempotency_key_hash, idempotency_request_hash) =
        idempotency_hashes(idempotency_key.as_deref(), request_hash);

    let _collection_lock =
        match acquire_file_lock(&collection_lock_path(&storage_root, &collection_name)) {
            Ok(lock) => lock,
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
        };

    let delete_req: IdListRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let delete_result = match with_collection_store(&storage_root, &collection_name, |store| {
        store
            .delete_idempotent(
                &delete_req.ids,
                idempotency_key_hash,
                idempotency_request_hash,
            )
            .map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    let payload = DeletePointsResponse {
        collection: collection_name.to_string(),
        deleted_count: delete_result.deleted_count,
        not_found_count: delete_result.not_found_count,
    };

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
    let fetched = match with_collection_store(&storage_root, &collection_name, |store| {
        store
            .fetch(&fetch_req.ids, include_values)
            .map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    let mut data = Vec::with_capacity(fetched.ids.len());
    for (idx, id) in fetched.ids.iter().enumerate() {
        let values = if include_values {
            fetched.vectors.as_ref().map(|all| {
                let start = idx * collection.dims as usize;
                let end = start + collection.dims as usize;
                all[start..end].to_vec()
            })
        } else {
            None
        };
        data.push(FetchedPoint {
            id: id.to_string(),
            values,
        });
    }

    let not_found: Vec<String> = fetched
        .missing_ids
        .iter()
        .map(ToString::to_string)
        .collect();

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

    let min_score = query_req.min_score;
    let approximate = query_req.approximate.unwrap_or(false);

    let mut flat_queries = Vec::with_capacity(queries.len() * collection.dims as usize);
    for query in &queries {
        flat_queries.extend_from_slice(query);
    }

    let search = match with_collection_store(&storage_root, &collection_name, |store| {
        store
            .search_batch_with_options(
                &flat_queries,
                collection.dims,
                top_k,
                collection.normalization == NormalizationPolicy::L2,
                approximate,
            )
            .map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    let mut results = Vec::with_capacity(queries.len());
    for qi in 0..queries.len() {
        let per_query = search.count_per_query as usize;
        let base = qi * per_query;
        let end = base + per_query;
        let mut matches = Vec::with_capacity(per_query);
        for idx in base..end {
            let score = search.scores[idx];
            if min_score.is_some_and(|min| score < min) {
                continue;
            }
            matches.push(QueryMatch {
                id: search.ids[idx].to_string(),
                score,
            });
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

    let stats = match with_collection_store(&storage_root, &collection_name, |store| {
        store.stats().map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    json_response(
        StatusCode::OK,
        &CollectionStatsResponse {
            collection: collection_name.to_string(),
            dims: collection.dims,
            metric: collection.metric,
            normalization: collection.normalization,
            visible_count: stats.visible_count,
            tombstone_count: stats.tombstone_count,
            segment_count: stats.segment_count,
            total_vector_count: stats.total_count,
            manifest_generation: stats.manifest_generation,
            index_ready_segments: stats.index_ready_segments,
            index_pending_segments: stats.index_pending_segments,
            index_failed_segments: stats.index_failed_segments,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/compact",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = CompactCollectionRequest,
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
    let compact_req: CompactCollectionRequest = if body.is_empty() {
        CompactCollectionRequest {
            expected_generation: None,
            ttl_max_age_ms: None,
            now_ms: None,
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
        }
    };
    let request_hash = hash_bytes(&body);
    let (idempotency_key_hash, idempotency_request_hash) =
        idempotency_hashes(idempotency_key.as_deref(), request_hash);

    let _collection_lock =
        match acquire_file_lock(&collection_lock_path(&storage_root, &collection_name)) {
            Ok(lock) => lock,
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
        };

    if compact_req.ttl_max_age_ms.is_some() && compact_req.expected_generation.is_some() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "compact supports either ttl_max_age_ms or expected_generation, not both",
        );
    }

    let compact = match with_collection_store(&storage_root, &collection_name, |store| {
        if let Some(ttl_max_age_ms) = compact_req.ttl_max_age_ms {
            let now_ms = compact_req.now_ms.unwrap_or_else(unix_ms_now);
            store
                .compact_expired_tombstones(collection.dims, now_ms, ttl_max_age_ms)
                .map_err(vector_error_response)
        } else if let Some(expected_generation) = compact_req.expected_generation {
            store
                .compact_with_generation(collection.dims, expected_generation)
                .map_err(vector_error_response)
        } else {
            store
                .compact_idempotent(
                    collection.dims,
                    idempotency_key_hash,
                    idempotency_request_hash,
                )
                .map_err(vector_error_response)
        }
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    let payload = CompactResponse {
        collection: collection_name.to_string(),
        dims: collection.dims,
        kept_count: compact.kept_count,
        removed_tombstones: compact.removed_tombstones,
    };

    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(
    post,
    path = "/v1/db/collections/{name}/indexes/build",
    tag = "DB",
    params(("name" = String, Path, description = "Collection name")),
    request_body = BuildIndexesRequest,
    responses((status = 200, body = BuildIndexesResponse))
)]
/// POST /v1/db/collections/{name}/indexes/build - Build pending ANN indexes.
pub async fn handle_build_collection_indexes(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let Some(collection_name) =
        extract_collection_name_with_suffix(req.uri().path(), "/indexes/build")
            .map(ToOwned::to_owned)
    else {
        return json_error(StatusCode::NOT_FOUND, "not_found", "Not found");
    };

    if let Err(resp) = require_collection(&storage_root, &collection_name).map(|_| ()) {
        return resp;
    }

    let body = match read_body(req, MAX_DB_REQUEST_BODY_BYTES).await {
        Ok(b) => b,
        Err(err) => return body_read_error_response(err),
    };
    if body.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "request body is required",
        );
    }
    let build_req: BuildIndexesRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };
    let max_segments = build_req
        .max_segments
        .unwrap_or(DEFAULT_INDEX_BUILD_MAX_SEGMENTS);
    if max_segments == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "max_segments must be greater than zero",
        );
    }

    let _collection_lock =
        match acquire_file_lock(&collection_lock_path(&storage_root, &collection_name)) {
            Ok(lock) => lock,
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &e),
        };

    let build = match with_collection_store(&storage_root, &collection_name, |store| {
        store
            .build_indexes_with_generation(build_req.expected_generation, max_segments)
            .map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };

    json_response(
        StatusCode::OK,
        &BuildIndexesResponse {
            collection: collection_name.to_string(),
            built_segments: build.built_segments,
            failed_segments: build.failed_segments,
            pending_segments: build.pending_segments,
        },
    )
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

    let changes = match with_collection_store(&storage_root, &collection_name, |store| {
        store.changes(since, limit).map_err(vector_error_response)
    }) {
        Ok(result) => result,
        Err(resp) => return resp,
    };
    let data: Vec<ChangeEventResponse> = changes
        .events
        .into_iter()
        .map(|event| ChangeEventResponse {
            seq: event.seq,
            op: match event.op {
                talu::vector::ChangeOp::Append => "append".to_string(),
                talu::vector::ChangeOp::Upsert => "upsert".to_string(),
                talu::vector::ChangeOp::Delete => "delete".to_string(),
                talu::vector::ChangeOp::Compact => "compact".to_string(),
            },
            id: event.id.to_string(),
            timestamp: event.timestamp,
        })
        .collect();

    json_response(
        StatusCode::OK,
        &ChangesResponse {
            collection: collection_name.to_string(),
            data,
            has_more: changes.has_more,
            next_since: changes.next_since,
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

fn collections_lock_path(storage_root: &Path) -> PathBuf {
    storage_root.join(COLLECTIONS_DIR).join(".collections.lock")
}

fn collection_store_root(storage_root: &Path, collection_name: &str) -> PathBuf {
    storage_root
        .join(COLLECTIONS_DIR)
        .join(COLLECTION_STORES_DIR)
        .join(collection_name)
}

fn collection_lock_path(storage_root: &Path, collection_name: &str) -> PathBuf {
    collection_store_root(storage_root, collection_name).join(".collection.lock")
}

struct FileLockGuard {
    _file: std::fs::File,
}

fn acquire_file_lock(path: &Path) -> Result<FileLockGuard, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed to create lock directory: {e}"))?;
    }

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("failed to open lock file: {e}"))?;

    file.lock()
        .map_err(|e| format!("failed to acquire file lock: {e}"))?;

    Ok(FileLockGuard { _file: file })
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

fn load_collections_cached(storage_root: &Path) -> Result<CollectionsDisk, String> {
    {
        let cache = COLLECTIONS_CACHE
            .read()
            .map_err(|_| "collections cache lock poisoned".to_string())?;
        if let Some(disk) = cache.get(storage_root) {
            return Ok(disk.clone());
        }
    }

    let mut cache = COLLECTIONS_CACHE
        .write()
        .map_err(|_| "collections cache lock poisoned".to_string())?;
    if let Some(disk) = cache.get(storage_root) {
        return Ok(disk.clone());
    }

    let disk = load_collections(storage_root)?;
    cache.insert(storage_root.to_path_buf(), disk.clone());
    Ok(disk)
}

fn store_collections_cache(storage_root: &Path, disk: &CollectionsDisk) -> Result<(), String> {
    let mut cache = COLLECTIONS_CACHE
        .write()
        .map_err(|_| "collections cache lock poisoned".to_string())?;
    cache.insert(storage_root.to_path_buf(), disk.clone());
    Ok(())
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
    let disk = load_collections_cached(storage_root)?;
    Ok(disk.collections.into_iter().find(|c| c.name == name))
}

fn with_collection_store<T>(
    storage_root: &Path,
    collection_name: &str,
    op: impl FnOnce(&VectorStore) -> Result<T, Response<BoxBody>>,
) -> Result<T, Response<BoxBody>> {
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
    op(&store)
}

fn clear_collection_store_cache(_storage_root: &Path, _collection_name: &str) {
    // Intentionally a no-op: vector-store caching is owned by core, not Rust bindings.
}

fn idempotency_key_from_headers(headers: &hyper::HeaderMap) -> Option<String> {
    headers
        .get("idempotency-key")
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned)
}

fn idempotency_hashes(key: Option<&str>, request_hash: u64) -> (u64, u64) {
    match key {
        Some(k) => (hash_bytes(k.as_bytes()), request_hash),
        None => (0, 0),
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

fn hash_bytes(data: &[u8]) -> u64 {
    // Stable FNV-1a 64-bit hash to keep idempotency keys deterministic
    // across Rust versions and process restarts.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
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
            if lower.contains("idempotencyconflict") || lower.contains("idempotency conflict") {
                json_error(
                    StatusCode::CONFLICT,
                    "idempotency_conflict",
                    "idempotency key was used with a different request payload",
                )
            } else if lower.contains("manifestgenerationconflict")
                || lower.contains("generation conflict")
            {
                json_error(
                    StatusCode::CONFLICT,
                    "generation_conflict",
                    "manifest generation does not match expected_generation",
                )
            } else if lower.contains("alreadyexists") || lower.contains("already exists") {
                json_error(
                    StatusCode::CONFLICT,
                    "id_conflict",
                    "append contains IDs that already exist; use upsert",
                )
            } else if lower.contains("dim") {
                json_error(StatusCode::BAD_REQUEST, "dimension_mismatch", &msg)
            } else {
                json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
            }
        }
        talu::VectorError::StoreError(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("idempotencyconflict") || lower.contains("idempotency conflict") {
                json_error(
                    StatusCode::CONFLICT,
                    "idempotency_conflict",
                    "idempotency key was used with a different request payload",
                )
            } else if lower.contains("manifestgenerationconflict")
                || lower.contains("generation conflict")
            {
                json_error(
                    StatusCode::CONFLICT,
                    "generation_conflict",
                    "manifest generation does not match expected_generation",
                )
            } else if lower.contains("alreadyexists") || lower.contains("already exists") {
                json_error(
                    StatusCode::CONFLICT,
                    "id_conflict",
                    "append contains IDs that already exist; use upsert",
                )
            } else if lower.contains("zerovectornotallowed") {
                json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_argument",
                    "zero vector cannot be L2-normalized",
                )
            } else if lower.contains("invalidcolumndata") || lower.contains("dim") {
                json_error(StatusCode::BAD_REQUEST, "dimension_mismatch", &msg)
            } else {
                json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
            }
        }
    }
}
