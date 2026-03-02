pub mod exec;
pub mod fs;
pub mod process;
pub mod shell;

use talu::policy::Policy;

use crate::server::state::AppState;

pub(crate) fn load_runtime_policy(state: &AppState) -> Result<Option<Policy>, String> {
    let Some(json) = state.agent_policy_json.as_deref() else {
        return Ok(None);
    };

    Policy::from_json(json)
        .map(Some)
        .map_err(|err| format!("failed to parse agent runtime policy JSON: {err}"))
}
