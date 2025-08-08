//! Utilities: logging init, paths, timing

use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

pub fn init_tracing() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init()
        .ok();
    Ok(())
}
