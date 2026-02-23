//! DB kv plane HTTP handlers.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};

use crate::server::auth_gateway::AuthContext;
use crate::server::repo;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    repo::handle_list_pins(state, req, auth).await
}

pub async fn handle_put(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    repo::handle_pin(state, req, auth).await
}

pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let Some(raw) = path.as_str().strip_prefix("/v1/db/kv/pins/") else {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("content-type", "application/json")
            .body(
                http_body_util::Full::new(Bytes::from_static(
                    br#"{"error":{"code":"invalid_path","message":"missing model id"}}"#,
                ))
                .boxed(),
            )
            .unwrap();
    };

    let model_id = percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
    repo::handle_unpin(state, req, auth, &model_id).await
}
