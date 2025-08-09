//! Pipeline orchestrator (stub)

use anyhow::Result;
use pcp_decoder::decode_to_pcm;
use pcp_fetcher::fetch_to_file;
use pcp_transcribe::{TranscribeOptions, transcribe_pcm};
use pcp_types::ProcessingResult;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub model: String,
    pub no_gpu: bool,
    pub language: String, // "auto" or ISO code
    pub gpu: String,      // cpu|metal|opencl
    pub progress: bool,
}

pub async fn run_one(
    source_url: &str,
    output_dir: &str,
    cfg: TranscriptionConfig,
) -> Result<ProcessingResult> {
    let start = Instant::now();
    let fetched = fetch_to_file(source_url, output_dir).await?;
    // Decode to 16k mono PCM (in-memory) for upcoming transcription stage
    let pcm = decode_to_pcm(&fetched.saved_path)?;
    // Transcribe with provided options
    let opts = TranscribeOptions {
        model: &cfg.model,
        no_gpu: cfg.no_gpu,
        language: Some(cfg.language.as_str()),
        gpu_backend: Some(cfg.gpu.as_str()),
    };
    let transcript = transcribe_pcm(&pcm, &opts)?;
    let processing_time = start.elapsed().as_secs_f64();

    Ok(ProcessingResult {
        source_url: fetched.final_url,
        audio_file_path: fetched.saved_path,
        transcript: transcript.text,
        summary: String::new(),
        processing_time,
    })
}
