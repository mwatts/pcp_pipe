//! Pipeline orchestrator (stub)

use anyhow::Result;
use pcp_fetcher::fetch_to_file;
use pcp_types::ProcessingResult;
use std::time::Instant;

pub async fn run_one(source_url: &str, output_dir: &str) -> Result<ProcessingResult> {
    let start = Instant::now();
    let fetched = fetch_to_file(source_url, output_dir).await?;
    let processing_time = start.elapsed().as_secs_f64();

    Ok(ProcessingResult {
        source_url: fetched.final_url,
        audio_file_path: fetched.saved_path,
        transcript: String::new(),
        summary: String::new(),
        processing_time,
    })
}
