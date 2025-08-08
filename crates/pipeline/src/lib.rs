//! Pipeline orchestrator (stub)

use anyhow::Result;
use pcp_types::ProcessingResult;

pub async fn run_one(source_url: &str, output_dir: &str) -> Result<ProcessingResult> {
    // Stubbed pipeline: returns an empty result without performing work.
    Ok(ProcessingResult {
        source_url: source_url.to_string(),
        audio_file_path: format!("{}/placeholder.dat", output_dir),
        transcript: String::new(),
        summary: String::new(),
        processing_time: 0.0,
    })
}
