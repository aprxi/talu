//! OpenResponses-compatible HTTP surface under `/v1/responses`.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use hyper::body::Incoming;
use hyper::{Request, Response};

use crate::server::auth_gateway::AuthContext;
use crate::server::handlers;
use crate::server::responses_types;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

#[utoipa::path(post, path = "/v1/responses", tag = "Responses",
    request_body = responses_types::CreateResponseBody,
    responses(
        (status = 200, body = responses_types::ResponseResource),
        (status = 400, body = responses_types::ResponsesErrorResponse, description = "Invalid request"),
        (status = 500, body = responses_types::ResponsesErrorResponse, description = "Generation failed"),
    ))]
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    handlers::handle_responses(state, req, auth_ctx).await
}
