//! Fetcher: placeholder API for downloading audio with Rust crates only.

use anyhow::{bail, Result};
use url::Url;

pub struct FetchResult {
    pub final_url: String,
    pub saved_path: String,
}

pub async fn fetch_to_file(_url: &str, _out_dir: &str) -> Result<FetchResult> {
    // Stub to keep workspace building. Implementation deferred to M1.
    let parsed = Url::parse(_url).map_err(|e| anyhow::anyhow!(e))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        bail!("unsupported scheme: {}", parsed.scheme());
    }
    Ok(FetchResult {
        final_url: parsed.into_string(),
        saved_path: format!("{}/placeholder.dat", _out_dir),
    })
}
