pub mod exec;
pub mod fs;
pub mod process;
pub mod shell;

use std::sync::Arc;

use talu::policy::Policy;

use crate::server::state::AppState;
use crate::server::{AgentRuntimeMode, SandboxBackend};

pub(crate) fn load_runtime_policy(state: &AppState) -> Option<Arc<Policy>> {
    state.agent_policy.clone()
}

pub(crate) fn runtime_mode_for_talu(mode: AgentRuntimeMode) -> talu::shell::AgentRuntimeMode {
    match mode {
        AgentRuntimeMode::Host => talu::shell::AgentRuntimeMode::Host,
        AgentRuntimeMode::Strict => talu::shell::AgentRuntimeMode::Strict,
    }
}

pub(crate) fn sandbox_backend_for_talu(backend: SandboxBackend) -> talu::shell::SandboxBackend {
    match backend {
        SandboxBackend::LinuxLocal => talu::shell::SandboxBackend::LinuxLocal,
        SandboxBackend::Oci => talu::shell::SandboxBackend::Oci,
        SandboxBackend::AppleContainer => talu::shell::SandboxBackend::AppleContainer,
    }
}
