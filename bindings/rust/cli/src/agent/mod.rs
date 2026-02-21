//! Agent subsystem for tool-calling modes.
//!
//! `talu agent` uses structured output to generate a single tool call,
//! asks the user for permission, executes it, and exits.

mod r#loop;
pub mod tools;

pub use r#loop::run_shell;

#[cfg(test)]
mod tools_test;
