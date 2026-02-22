mod ask;
mod convert;
mod db;
mod file;
mod models;
mod repo;
mod sessions;
mod shell;
mod util;
mod xray;

use std::env;
use std::io::{self, IsTerminal};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};

use crate::server::{run_server, ServerArgs};

use talu::LogFormat;
use talu::LogLevel;

/// Log output format (CLI argument)
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(super) enum CliLogFormat {
    /// Human-readable colored output
    Human,
    /// JSON (OpenTelemetry compliant)
    Json,
}

#[derive(Parser)]
#[command(
    name = "talu",
    about = "High-performance LLM inference library",
    version = env!("TALU_VERSION")
)]
pub(super) struct Cli {
    /// Increase verbosity (-v=info, -vv=debug, -vvv=trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Log output format (default: human for TTY, json for pipes)
    #[arg(long, value_enum, global = true)]
    pub log_format: Option<CliLogFormat>,

    /// Filter log output by scope (glob pattern, e.g. "core::*", "server::*", "core::inference")
    #[arg(long, global = true)]
    pub log_filter: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub(super) enum Commands {
    /// Show help
    Help,
    /// Chat (new session)
    Ask(AskArgs),
    /// Run the HTTP server
    Serve(ServerArgs),
    /// Convert/quantize model
    Convert(ConvertArgs),
    /// Tokenize text
    Tokenize(TokenizeArgs),
    /// List cached models or files in a model
    Ls(LsArgs),
    /// Download model to cache (or browse HuggingFace interactively)
    Get(GetArgs),
    /// Sample pinned models with inference
    Sample(SampleArgs),
    /// Remove model from cache
    Rm(RmArgs),
    /// Describe model architecture and execution plan
    Describe(DescribeArgs),
    /// XRay kernel profiling
    Xray(XrayArgs),
    /// Agent mode (tool-calling loop with execute_command)
    Agent(AgentArgs),
    /// Set default model (interactive picker or explicit)
    Set(SetArgs),
    /// Inspect and transform input files for inference
    File(FileArgs),
}

#[derive(Args)]
pub(super) struct DbArgs {
    #[command(subcommand)]
    pub command: DbCommands,
}

#[derive(Subcommand)]
pub(super) enum DbCommands {
    /// Initialize a new TaluDB storage directory
    Init(DbInitArgs),
    /// List sessions in a TaluDB storage
    List(DbListArgs),
    /// Show details of a specific session
    Show(DbShowArgs),
    /// Delete a session from storage
    Delete(DbDeleteArgs),
}

#[derive(Args)]
pub(super) struct DbInitArgs {
    /// Path for storage directory (default: ./taludb)
    pub path: Option<std::path::PathBuf>,

    /// Import existing store.key from another database
    #[arg(long)]
    pub import_key: Option<std::path::PathBuf>,
}

#[derive(Args)]
pub(super) struct DbListArgs {
    /// Path to TaluDB storage (default: ./taludb)
    pub path: Option<std::path::PathBuf>,

    /// Maximum sessions to display (default: 50, use 0 for unlimited)
    #[arg(short = 'n', long)]
    pub limit: Option<usize>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: DbOutputFormat,
}

#[derive(clap::ValueEnum, Clone, Copy)]
pub(super) enum DbOutputFormat {
    Table,
    Json,
    Csv,
}

#[derive(Args)]
pub(super) struct DbShowArgs {
    /// Session ID to show
    pub session_id: String,

    /// Path to TaluDB storage (default: ./taludb)
    pub path: Option<std::path::PathBuf>,

    /// Output format
    #[arg(long, value_enum, default_value = "pretty")]
    pub format: DbShowFormat,

    /// Show raw JSON for debugging
    #[arg(long)]
    pub raw: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, PartialEq)]
pub(super) enum DbShowFormat {
    Pretty,
    Json,
}

#[derive(Args)]
pub(super) struct DbDeleteArgs {
    /// Session ID to delete
    pub session_id: String,

    /// Path to TaluDB storage (default: ./taludb)
    pub path: Option<std::path::PathBuf>,

    /// Skip confirmation prompt
    #[arg(short, long)]
    pub force: bool,
}

#[derive(Args)]
pub(super) struct SetArgs {
    /// Model ID to set as default, or "show" to display config
    pub model: Option<String>,

    /// Set default model URI and print it only (script-friendly)
    #[arg(long = "model-uri")]
    pub model_uri: Option<String>,
}

#[derive(Args)]
pub(super) struct AskArgs {
    /// Model to use (overrides MODEL_URI env and `talu set` default)
    #[arg(
        short = 'm',
        long = "model-uri",
        alias = "model_uri",
        env = "MODEL_URI"
    )]
    pub model: Option<String>,

    /// Override default endpoint URL for remote providers
    #[arg(long)]
    pub endpoint_url: Option<String>,

    /// System message for chat mode
    #[arg(short = 'S', long, default_value = "You are a helpful assistant.")]
    pub system: String,

    /// Skip chat template, use raw prompt directly
    #[arg(long)]
    pub no_chat: bool,

    /// Disable streaming output
    #[arg(long)]
    pub no_stream: bool,

    /// Show raw output (no stripping of special tokens)
    #[arg(long)]
    pub raw: bool,

    /// Hide reasoning/thinking content from output
    #[arg(long, conflicts_with = "raw")]
    pub hide_thinking: bool,

    /// Random seed for deterministic generation (0 = random)
    #[arg(long, env = "SEED")]
    pub seed: Option<u64>,

    /// Storage profile name
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path (bypasses profile resolution)
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,

    /// Disable storage entirely
    #[arg(long)]
    pub no_bucket: bool,

    /// Suppress progress bars during model loading
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Create a new session and output its UUID (no prompt allowed)
    #[arg(long)]
    pub new: bool,

    /// Create a new session and output its ID only (requires prompt)
    #[arg(long = "session-id")]
    pub session_id_only: bool,

    /// Append to an existing session (UUID or prefix)
    #[arg(long)]
    pub session: Option<String>,

    /// No stdout output (errors still go to stderr)
    #[arg(short = 's', long)]
    pub silent: bool,

    /// Delete an existing session (requires --session)
    #[arg(long)]
    pub delete: bool,

    /// Output format (json only)
    #[arg(long, value_enum)]
    pub format: Option<AskOutputFormat>,

    /// Write output to file instead of stdout
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Text prompt
    #[arg(trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AskOutputFormat {
    Json,
}

#[derive(Args)]
pub(super) struct AgentArgs {
    /// Model to use (overrides MODEL_URI env and `talu set` default)
    #[arg(short = 'm', long = "model", env = "MODEL_URI")]
    pub model: Option<String>,

    /// Random seed for deterministic generation (0 = random)
    #[arg(long, env = "SEED")]
    pub seed: Option<u64>,

    /// Storage profile name
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path (bypasses profile resolution)
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,

    /// Disable storage entirely
    #[arg(long)]
    pub no_bucket: bool,

    /// Path to a JSON policy file for tool call filtering (IAM-style).
    /// Overrides the built-in default-deny shell policy.
    #[arg(long, env = "TOOL_POLICY")]
    pub policy: Option<String>,

    /// Path to a custom tool manifest file or a directory of manifests.
    /// Can be provided multiple times.
    #[arg(long, value_name = "PATH")]
    pub tools: Vec<String>,

    /// Timeout in seconds for custom tools.
    #[arg(long, value_name = "SECONDS", default_value_t = 60)]
    pub tool_timeout: u64,

    /// Text prompt
    #[arg(trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Args)]
pub(super) struct ConvertArgs {
    /// Model: local path or HuggingFace model ID (e.g., Qwen/Qwen3-0.6B)
    pub model: Option<String>,

    /// Quantization scheme (use --scheme help to list available)
    #[arg(long, default_value = "gaf4_64", value_name = "SCHEME", num_args = 1)]
    pub scheme: String,

    /// Output directory (default: $TALU_HOME/models)
    #[arg(long)]
    pub output: Option<String>,

    /// Overwrite existing output
    #[arg(short, long)]
    pub force: bool,

    /// Output JSON: {"success": true, "output_path": "..."}
    #[arg(long)]
    pub json: bool,

    /// Output converted model URI only (script-friendly)
    #[arg(long = "model-uri")]
    pub model_uri_only: bool,

    /// Output only the path (for scripting)
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Args)]
pub(super) struct TokenizeArgs {
    /// Model: local path or HuggingFace model ID
    pub model: String,

    /// Text to tokenize
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FileFitArg {
    Stretch,
    Contain,
    Cover,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FileFormatArg {
    Jpeg,
    Png,
}

#[derive(Args)]
pub(super) struct FileArgs {
    /// Input file path (reads from stdin if omitted or "-")
    pub path: Option<PathBuf>,

    /// Resize to WxH (e.g. 512x512)
    #[arg(long)]
    pub resize: Option<String>,

    /// Fit mode when resizing
    #[arg(long, value_enum)]
    pub fit: Option<FileFitArg>,

    /// Output file path (never overwrites input path)
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Output format (jpeg|png). If omitted, inferred from output extension or input.
    #[arg(long, value_enum)]
    pub format: Option<FileFormatArg>,

    /// JPEG quality (1-100) when output format is jpeg
    #[arg(long)]
    pub quality: Option<u8>,
}

#[derive(Args)]
pub(super) struct LsArgs {
    /// Target: provider:: (vllm::, ollama::), Org prefix (Qwen), or Org/Model for files
    pub target: Option<String>,

    /// Override default endpoint URL for remote providers
    #[arg(long)]
    pub endpoint_url: Option<String>,

    /// Show only quantized managed models (~/.cache/talu/models)
    #[arg(short = 'Q', long, conflicts_with = "hub_only")]
    pub quantized_only: bool,

    /// Show only HuggingFace hub models (~/.cache/huggingface/hub)
    #[arg(short = 'H', long, conflicts_with = "quantized_only")]
    pub hub_only: bool,

    /// Show only pinned models for the active profile
    #[arg(short = 'P', long)]
    pub pinned: bool,

    /// Storage profile name for pinned model metadata
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path for pinned model metadata
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,
}

#[derive(Args)]
pub(super) struct GetArgs {
    /// Org/Model (omit for interactive search)
    pub target: Option<String>,

    /// Force re-download even if cached
    #[arg(short, long)]
    pub force: bool,

    /// Output model URI only (script-friendly)
    #[arg(long = "model-uri")]
    pub model_uri_only: bool,

    /// Custom HF endpoint URL (overrides HF_ENDPOINT env var)
    #[arg(long, env = "HF_ENDPOINT")]
    pub endpoint_url: Option<String>,

    /// Storage profile name for pinned model metadata
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path for pinned model metadata
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,

    /// Add a model URI to the pinned list for this profile
    #[arg(
        long,
        value_name = "MODEL_URI",
        conflicts_with_all = ["remove_pin", "sync_pins", "no_dry_run"]
    )]
    pub add_pin: Option<String>,

    /// Remove a model URI from the pinned list for this profile
    #[arg(
        long,
        value_name = "MODEL_URI",
        conflicts_with_all = ["add_pin", "sync_pins", "no_dry_run"]
    )]
    pub remove_pin: Option<String>,

    /// Sync pinned model URIs for this profile (download missing cached models)
    #[arg(long)]
    pub sync_pins: bool,

    /// Disable dry-run for --sync-pins and perform downloads
    #[arg(long, requires = "sync_pins")]
    pub no_dry_run: bool,

    /// Skip weight files when syncing pins (download only metadata/tokenizer)
    #[arg(long, requires = "sync_pins")]
    pub no_weights: bool,
}

#[derive(Args)]
pub(super) struct SampleArgs {
    /// Storage profile name for pinned model metadata
    #[arg(long, env = "TALU_PROFILE", default_value = "default")]
    pub profile: String,

    /// Override bucket path for pinned model metadata
    #[arg(long, env = "TALU_BUCKET")]
    pub bucket: Option<PathBuf>,

    /// Maximum pinned models to sample (newest pins first)
    #[arg(long)]
    pub max_models: Option<usize>,
}

#[derive(Args)]
pub(super) struct RmArgs {
    /// Org/Model (one or more)
    #[arg(required = true)]
    pub targets: Vec<String>,

    /// Script-friendly mode: no stdout output
    #[arg(long = "model-uri")]
    pub model_uri_only: bool,
}

#[derive(Args)]
pub(super) struct DescribeArgs {
    /// Model: local path or HuggingFace model ID
    pub model: String,

    /// Show execution plan (kernels that will be used)
    #[arg(long)]
    pub plan: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub(super) struct XrayArgs {
    /// Model: local path or HuggingFace model ID
    pub model: String,

    /// Profile input processing (prefill). This is the default.
    #[arg(long, group = "xray_mode")]
    pub input: bool,

    /// Profile output generation (decode, single token step)
    #[arg(long, group = "xray_mode")]
    pub output: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Prompt text (default: "xray")
    #[arg(trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

/// Configure log level based on verbosity flags
fn configure_logging(verbose: u8, log_format: Option<CliLogFormat>, log_filter: Option<&str>) {
    // Set log level based on -v count
    let level = match verbose {
        0 => LogLevel::Warn,  // Default - silent for normal operation
        1 => LogLevel::Info,  // -v
        2 => LogLevel::Debug, // -vv
        _ => LogLevel::Trace, // -vvv or more
    };
    talu::logging::set_log_level(level);

    // Set log format if explicitly specified
    if let Some(format) = log_format {
        let talu_format = match format {
            CliLogFormat::Json => LogFormat::Json,
            CliLogFormat::Human => LogFormat::Human,
        };
        talu::logging::set_log_format(talu_format);
    }

    // Set log filter for Zig core logs
    if let Some(filter) = log_filter {
        talu::logging::set_log_filter(filter);
    }
}

/// Check if we should implicitly add "ask" subcommand
/// This handles: `talu -m model prompt` and `echo prompt | talu -m model`
fn should_implicit_ask(args: &[String], stdin_is_pipe: bool) -> bool {
    // If no args beyond program name, no implicit
    if args.len() <= 1 {
        return stdin_is_pipe; // stdin pipe with no args -> implicit ask
    }

    let first = &args[1];

    // If first arg is a known subcommand, don't add implicit ask
    let subcommands = [
        "ask", "help", "serve", "convert", "tokenize", "ls", "get", "rm", "describe", "xray",
        "agent", "set", "sample", "file",
    ];
    if subcommands.iter().any(|&cmd| cmd == first) {
        return false;
    }

    // If first arg is -h/--help/-V/--version, don't add implicit
    if first == "-h" || first == "--help" || first == "-V" || first == "--version" {
        return false;
    }

    // If first arg starts with - (a flag like -m or -v), implicit ask
    if first.starts_with('-') {
        return true;
    }

    // If stdin is piped, implicit ask
    if stdin_is_pipe {
        return true;
    }

    false
}

fn run_inner() -> Result<()> {
    let args = collect_args();
    let stdin_is_pipe = !io::stdin().is_terminal();

    let parsed = if should_implicit_ask(&args, stdin_is_pipe) {
        // Implicit ask: `talu -m model prompt` -> `talu ask -m model prompt`
        let mut implicit = Vec::with_capacity(args.len() + 1);
        implicit.push("talu".to_string());
        implicit.push("ask".to_string());
        implicit.extend(args.iter().skip(1).cloned());
        Cli::parse_from(implicit)
    } else {
        Cli::parse_from(&args)
    };

    // Configure logging before any operations
    configure_logging(
        parsed.verbose,
        parsed.log_format,
        parsed.log_filter.as_deref(),
    );

    match parsed.command {
        None => {
            print_usage();
            Ok(())
        }
        Some(Commands::Help) => {
            print_usage();
            Ok(())
        }
        Some(Commands::Ask(args)) => ask::cmd_ask(args, stdin_is_pipe, parsed.verbose),
        Some(Commands::Serve(args)) => {
            run_server(args, parsed.verbose, parsed.log_filter.as_deref())
        }
        Some(Commands::Tokenize(args)) => convert::cmd_tokenize(args),
        Some(Commands::Convert(args)) => convert::cmd_convert(args),
        Some(Commands::Ls(args)) => models::cmd_ls(args),
        Some(Commands::Get(args)) => repo::cmd_get(args),
        Some(Commands::Sample(args)) => repo::cmd_sample(args),
        Some(Commands::Rm(args)) => repo::cmd_rm(args),
        Some(Commands::Describe(args)) => models::cmd_describe(args),
        Some(Commands::Xray(args)) => xray::cmd_xray(args),
        Some(Commands::Agent(args)) => shell::cmd_agent(args, stdin_is_pipe),
        Some(Commands::Set(args)) => cmd_set(args),
        Some(Commands::File(args)) => file::cmd_file(args),
    }
}

fn cmd_set(args: SetArgs) -> Result<()> {
    if let Some(model_uri) = args.model_uri.as_deref() {
        if args.model.is_some() {
            bail!("Error: cannot combine positional model with --model-uri.");
        }
        crate::config::set_default_model(model_uri)?;
        println!("{model_uri}");
        return Ok(());
    }

    match args.model.as_deref() {
        // talu set show — display current config
        Some("show") => {
            let config = crate::config::load_config()?;
            let path = crate::config::config_path();
            println!("Config: {}", path.display());
            println!();
            match &config.default_model {
                Some(model) => println!("  default_model = {}", model),
                None => println!("  default_model = (not set)"),
            }
            if !config.profiles.is_empty() {
                println!();
                for (name, profile) in &config.profiles {
                    println!("  [profiles.{}]", name);
                    println!("    bucket = {}", profile.bucket.display());
                }
            }
            Ok(())
        }
        // talu set Qwen/Qwen3-0.6B — set directly
        Some(model) => {
            crate::config::set_default_model(model)?;
            eprintln!();
            eprintln!(
                "  \x1b[1;32m✓\x1b[0m  Default model set to \x1b[1;37m{}\x1b[0m",
                model
            );
            eprintln!();
            Ok(())
        }
        // talu set — interactive picker
        None => {
            match crate::model_selector::run_interactive_selector()? {
                Some(model) => {
                    eprintln!();
                    eprintln!(
                        "  \x1b[1;32m✓\x1b[0m  Default model set to \x1b[1;37m{}\x1b[0m",
                        model
                    );
                    eprintln!();
                }
                None => eprintln!("No changes."),
            }
            Ok(())
        }
    }
}

// Temporarily disabled: db subcommand is intentionally not exposed in Commands.
#[allow(dead_code)]
fn cmd_db(args: DbArgs) -> Result<()> {
    match args.command {
        DbCommands::Init(args) => db::cmd_db_init(args),
        DbCommands::List(args) => db::cmd_db_list(args),
        DbCommands::Show(args) => db::cmd_db_show(args),
        DbCommands::Delete(args) => db::cmd_db_delete(args),
    }
}

pub fn run() -> i32 {
    if let Err(err) = run_inner() {
        eprintln!("{err}");
        1
    } else {
        0
    }
}

fn collect_args() -> Vec<String> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let args: Vec<String> = env::args_os()
            .map(|arg| String::from_utf8_lossy(arg.as_os_str().as_bytes()).to_string())
            .collect();
        normalize_args(args)
    }
    #[cfg(not(unix))]
    {
        let args: Vec<String> = env::args().collect();
        normalize_args(args)
    }
}

fn normalize_args(args: Vec<String>) -> Vec<String> {
    args
}

fn print_usage() {
    let mut cmd = Cli::command();
    let _ = cmd.print_help();
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, FromArgMatches};

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut cmd = Cli::command();
        cmd = cmd.disable_help_subcommand(true);
        let matches = cmd.try_get_matches_from(args)?;
        Cli::from_arg_matches(&matches)
    }

    #[test]
    fn parse_ask_with_session_target() {
        let cli =
            parse(&["talu", "ask", "--session", "abc123", "hello"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert_eq!(args.session.as_deref(), Some("abc123"));
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn reject_removed_ask_flags() {
        assert!(parse(&["talu", "ask", "-r", "1"]).is_err());
    }

    #[test]
    fn parse_ask_delete_with_session() {
        let cli = parse(&["talu", "ask", "--session", "abc123", "--delete"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert_eq!(args.session.as_deref(), Some("abc123"));
                assert!(args.delete);
                assert!(args.prompt.is_empty());
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_ask_session_id_flag() {
        let cli = parse(&["talu", "ask", "--session-id", "hello"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert!(args.session_id_only);
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_ask_silent_flag() {
        let cli = parse(&["talu", "ask", "-s", "hello"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert!(args.silent);
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_ask_raw_flag() {
        let cli = parse(&["talu", "ask", "--raw", "hello"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert!(args.raw);
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_ask_hide_thinking_flag() {
        let cli =
            parse(&["talu", "ask", "--hide-thinking", "hello"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert!(args.hide_thinking);
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn reject_ask_hide_thinking_with_raw() {
        assert!(parse(&["talu", "ask", "--raw", "--hide-thinking", "hello"]).is_err());
    }

    #[test]
    fn parse_ask_model_uri_flag() {
        let cli = parse(&["talu", "ask", "--model-uri", "Qwen/Qwen3-0.6B", "hello"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert_eq!(args.model.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_ask_model_uri_underscore_alias() {
        let cli = parse(&["talu", "ask", "--model_uri", "Qwen/Qwen3-0.6B", "hello"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Ask(args)) => {
                assert_eq!(args.model.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert_eq!(args.prompt, vec!["hello"]);
            }
            _ => panic!("expected ask command"),
        }
    }

    #[test]
    fn parse_rm_model_uri_flag() {
        let cli =
            parse(&["talu", "rm", "Qwen/Qwen3-0.6B", "--model-uri"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Rm(args)) => {
                assert_eq!(args.targets, vec!["Qwen/Qwen3-0.6B"]);
                assert!(args.model_uri_only);
            }
            _ => panic!("expected rm command"),
        }
    }

    #[test]
    fn parse_set_model_uri_flag() {
        let cli = parse(&["talu", "set", "--model-uri", "Qwen/Qwen3-0.6B"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Set(args)) => {
                assert_eq!(args.model_uri.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert!(args.model.is_none());
            }
            _ => panic!("expected set command"),
        }
    }

    #[test]
    fn parse_get_profile_and_bucket_flags() {
        let cli = parse(&[
            "talu",
            "get",
            "--profile",
            "work",
            "--bucket",
            "/tmp/talu-bucket",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Get(args)) => {
                assert_eq!(args.profile, "work");
                assert_eq!(args.bucket, Some(PathBuf::from("/tmp/talu-bucket")));
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn parse_get_sync_pins_flag() {
        let cli = parse(&["talu", "get", "--sync-pins"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Get(args)) => {
                assert!(args.sync_pins);
                assert!(!args.no_dry_run);
                assert!(args.add_pin.is_none());
                assert!(args.remove_pin.is_none());
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn parse_get_sync_pins_no_dry_run_flag() {
        let cli =
            parse(&["talu", "get", "--sync-pins", "--no-dry-run"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Get(args)) => {
                assert!(args.sync_pins);
                assert!(args.no_dry_run);
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn reject_get_no_dry_run_without_sync_pins() {
        assert!(parse(&["talu", "get", "--no-dry-run"]).is_err());
    }

    #[test]
    fn parse_get_add_pin_flag() {
        let cli =
            parse(&["talu", "get", "--add-pin", "Qwen/Qwen3-0.6B"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Get(args)) => {
                assert_eq!(args.add_pin.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert!(args.remove_pin.is_none());
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn parse_get_remove_pin_flag() {
        let cli = parse(&["talu", "get", "--remove-pin", "Qwen/Qwen3-0.6B"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Get(args)) => {
                assert_eq!(args.remove_pin.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert!(args.add_pin.is_none());
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn reject_get_add_and_remove_pin_flags() {
        assert!(parse(&[
            "talu",
            "get",
            "--add-pin",
            "Qwen/Qwen3-0.6B",
            "--remove-pin",
            "Qwen/Qwen3-0.6B"
        ])
        .is_err());
    }

    #[test]
    fn parse_sample_max_models_flag() {
        let cli = parse(&["talu", "sample", "--max-models", "10"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Sample(args)) => {
                assert_eq!(args.max_models, Some(10));
                assert_eq!(args.profile, "default");
            }
            _ => panic!("expected sample command"),
        }
    }

    #[test]
    fn parse_sample_profile_and_bucket_flags() {
        let cli = parse(&[
            "talu",
            "sample",
            "--profile",
            "work",
            "--bucket",
            "/tmp/talu-bucket",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Sample(args)) => {
                assert_eq!(args.profile, "work");
                assert_eq!(args.bucket, Some(PathBuf::from("/tmp/talu-bucket")));
            }
            _ => panic!("expected sample command"),
        }
    }

    #[test]
    fn parse_ls_quantized_only_flag() {
        let cli = parse(&["talu", "ls", "-Q"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ls(args)) => {
                assert!(args.quantized_only);
                assert!(!args.hub_only);
                assert!(!args.pinned);
            }
            _ => panic!("expected ls command"),
        }
    }

    #[test]
    fn parse_ls_hub_only_flag() {
        let cli = parse(&["talu", "ls", "-H"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ls(args)) => {
                assert!(args.hub_only);
                assert!(!args.quantized_only);
                assert!(!args.pinned);
            }
            _ => panic!("expected ls command"),
        }
    }

    #[test]
    fn parse_ls_pinned_flag() {
        let cli = parse(&["talu", "ls", "-P"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Ls(args)) => {
                assert!(args.pinned);
                assert!(!args.quantized_only);
                assert!(!args.hub_only);
            }
            _ => panic!("expected ls command"),
        }
    }

    #[test]
    fn reject_ls_source_flag_conflict() {
        assert!(parse(&["talu", "ls", "-Q", "-H"]).is_err());
    }

    #[test]
    fn parse_ls_pinned_with_quantized_flag() {
        let cli = parse(&["talu", "ls", "-P", "-Q"]).expect("parse should succeed");
        match cli.command {
            Some(Commands::Ls(args)) => {
                assert!(args.pinned);
                assert!(args.quantized_only);
            }
            _ => panic!("expected ls command"),
        }
    }

    #[test]
    fn parse_ls_pinned_with_hub_flag() {
        let cli = parse(&["talu", "ls", "-P", "-H"]).expect("parse should succeed");
        match cli.command {
            Some(Commands::Ls(args)) => {
                assert!(args.pinned);
                assert!(args.hub_only);
            }
            _ => panic!("expected ls command"),
        }
    }

    #[test]
    fn parse_file_info_command() {
        let cli = parse(&["talu", "file", "image.png"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::File(args)) => {
                assert_eq!(args.path, Some(PathBuf::from("image.png")));
                assert!(args.resize.is_none());
                assert!(args.output.is_none());
            }
            _ => panic!("expected file command"),
        }
    }

    #[test]
    fn parse_file_transform_flags() {
        let cli = parse(&[
            "talu",
            "file",
            "image.png",
            "--resize",
            "512x512",
            "--fit",
            "cover",
            "--output",
            "out.png",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::File(args)) => {
                assert_eq!(args.resize.as_deref(), Some("512x512"));
                assert_eq!(args.fit, Some(FileFitArg::Cover));
                assert_eq!(args.output, Some(PathBuf::from("out.png")));
            }
            _ => panic!("expected file command"),
        }
    }

    #[test]
    fn parse_agent_command() {
        let cli = parse(&[
            "talu",
            "agent",
            "--model",
            "Qwen/Qwen3-0.6B",
            "list",
            "files",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Agent(args)) => {
                assert_eq!(args.model.as_deref(), Some("Qwen/Qwen3-0.6B"));
                assert_eq!(args.prompt, vec!["list", "files"]);
            }
            _ => panic!("expected agent command"),
        }
    }

    #[test]
    fn reject_shell_subcommand() {
        assert!(parse(&["talu", "shell", "--model", "Qwen/Qwen3-0.6B", "pwd"]).is_err());
    }
}
