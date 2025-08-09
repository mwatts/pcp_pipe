//! Pipeline orchestrator

use anyhow::Result;
use pcp_fetcher::fetch_to_file;
use pcp_types::ProcessingResult;
use pcp_transcribe::{NativeTranscribeOptions, transcribe_file_async};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub language: String, // "auto" or ISO code
}

pub async fn run_one(
    source_url: &str,
    output_dir: &str,
    cfg: TranscriptionConfig,
) -> Result<ProcessingResult> {
    let start = Instant::now();
    let fetched = fetch_to_file(source_url, output_dir).await?;

    // Use native macOS Speech framework for direct file transcription
    tracing::info!("Using native macOS Speech framework for transcription");
    let native_opts = NativeTranscribeOptions {
        locale: if cfg.language == "auto" { None } else { Some(cfg.language.clone()) },
        on_device: true, // Prefer on-device processing
        report_partials: false, // Only final results for now
    };
    let transcript = transcribe_file_async(std::path::Path::new(&fetched.saved_path), &native_opts).await?;

    let processing_time = start.elapsed().as_secs_f64();

    Ok(ProcessingResult {
        source_url: fetched.final_url,
        audio_file_path: fetched.saved_path,
        transcript: transcript.text,
        summary: String::new(),
        processing_time,
    })
}
