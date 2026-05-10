mod cli;
pub mod config;
mod provider;
pub(crate) mod quant_scheme;
pub mod server;

#[no_mangle]
pub extern "C" fn talu_cli_main() -> i32 {
    cli::run()
}
