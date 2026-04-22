pub mod exec;
pub mod fs;
pub mod process;
pub mod shell;

use std::sync::Arc;

use talu::{
    AgentRuntimeMode as TaluAgentRuntimeMode, Policy, SandboxBackend as TaluSandboxBackend,
};

use crate::server::state::AppState;
use crate::server::{AgentRuntimeMode, SandboxBackend};

pub(crate) fn load_runtime_policy(state: &AppState) -> Option<Arc<Policy>> {
    state.agent_policy.clone()
}

pub(crate) fn runtime_mode_for_talu(state: &AppState) -> TaluAgentRuntimeMode {
    match state.agent_runtime_mode {
        AgentRuntimeMode::Host => TaluAgentRuntimeMode::Host,
        AgentRuntimeMode::Strict => TaluAgentRuntimeMode::Strict,
    }
}

pub(crate) fn sandbox_backend_for_talu(backend: SandboxBackend) -> TaluSandboxBackend {
    match backend {
        SandboxBackend::LinuxLocal => TaluSandboxBackend::LinuxLocal,
        SandboxBackend::Oci => TaluSandboxBackend::Oci,
        SandboxBackend::AppleContainer => TaluSandboxBackend::AppleContainer,
    }
}

pub(crate) fn default_workdir(state: &AppState) -> Option<String> {
    state
        .workdir
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned())
}

pub(crate) fn compute_accept_key(key: &[u8]) -> String {
    tokio_tungstenite::tungstenite::handshake::derive_accept_key(key)
}
