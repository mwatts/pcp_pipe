//! Pipeline orchestrator (stub)

use anyhow::Result;
use pcp_decoder::decode_to_pcm;
use pcp_fetcher::fetch_to_file;
use pcp_types::ProcessingResult;
use std::time::Instant;

pub async fn run_one(source_url: &str, output_dir: &str) -> Result<ProcessingResult> {
    let start = Instant::now();
    let fetched = fetch_to_file(source_url, output_dir).await?;
    // Decode to 16k mono PCM (in-memory) for upcoming transcription stage
    let _pcm = decode_to_pcm(&fetched.saved_path)?;
    let processing_time = start.elapsed().as_secs_f64();

    Ok(ProcessingResult {
        source_url: fetched.final_url,
        audio_file_path: fetched.saved_path,
        transcript: String::new(),
        summary: String::new(),
        processing_time,
    })
}
