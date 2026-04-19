mod cli;
pub mod config;
pub(crate) mod model_selector;
mod provider;
pub(crate) mod quant_scheme;
pub mod server;
pub(crate) mod tui_common;

#[no_mangle]
pub extern "C" fn talu_cli_main() -> i32 {
    cli::run()
}
