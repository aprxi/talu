mod agent;
pub mod bucket_settings;
mod cli;
pub mod config;
pub(crate) mod hf;
pub(crate) mod model_selector;
pub(crate) mod pin_store;
mod provider;
pub mod server;
pub(crate) mod tui_common;

#[no_mangle]
pub extern "C" fn talu_cli_main() -> i32 {
    cli::run()
}
