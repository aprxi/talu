use anyhow::{bail, Result};

use super::SqlArgs;
use talu::SqlEngine;

pub(super) fn cmd_sql(args: SqlArgs) -> Result<()> {
    let db_root = if let Some(explicit) = args.bucket {
        explicit
    } else {
        crate::config::resolve_and_ensure_bucket(&args.profile)?
    };

    if args.no_bucket {
        bail!("Error: --no-bucket is not supported for SQL queries.");
    }

    let db_root_str = db_root.to_string_lossy();
    let result = SqlEngine::query_json(&db_root_str, &args.query)?;
    println!("{result}");
    Ok(())
}
