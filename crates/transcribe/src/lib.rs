use anyhow::Result;
use pcp_decoder::PcmFrames;
use tracing::instrument;

// Kalosm audio transcription
use kalosm::sound::rodio::buffer::SamplesBuffer;
use kalosm::sound::{
    AsyncSourceTranscribeExt, TextStream, Whisper, WhisperLanguage, WhisperSource,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transcript {
    pub text: String,
}

pub struct TranscribeOptions<'a> {
    pub model: &'a str, // maps to kalosm::sound::WhisperSource
    pub no_gpu: bool,
    pub language: Option<&'a str>, // Some("auto") for autodetect or specific ISO code
    pub gpu_backend: Option<&'a str>, // unused in Kalosm (kept for compatibility)
}

#[instrument(skip(pcm, opts))]
pub async fn transcribe_pcm_async(
    pcm: &PcmFrames,
    opts: &TranscribeOptions<'_>,
) -> Result<Transcript> {
    // Build a rodio SamplesBuffer from in-memory PCM (mono, 16k)
    let channels = pcm.channels;
    let sample_rate = pcm.sample_rate;
    let data = pcm.samples.clone();
    let source = SamplesBuffer::new(channels, sample_rate, data);

    // Resolve model and language
    let whisper_src = parse_whisper_source(opts.model)?;
    let language = parse_language_opt(opts.language)?;

    // Load model (downloads/cache handled internally by Kalosm)
    let mut builder = Whisper::builder().with_source(whisper_src);
    builder = builder.with_language(language);
    let model = builder.build().await?;

    // Transcribe and collect full text
    let mut stream = source.transcribe(model);
    let text = stream.all_text().await;
    Ok(Transcript { text })
}

// Optional compatibility wrapper for existing callers; blocks current thread to run async
#[instrument(skip(pcm, opts))]
pub fn transcribe_pcm(pcm: &PcmFrames, opts: &TranscribeOptions<'_>) -> Result<Transcript> {
    tokio::runtime::Handle::try_current().map_err(|_| {
        anyhow::anyhow!("no Tokio runtime; use transcribe_pcm_async in async context")
    })?;
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(transcribe_pcm_async(pcm, opts))
    })
}

// -------- Model management API --------

// Parse model source string into Kalosm WhisperSource; default to a quantized model
fn parse_whisper_source(s: &str) -> Result<WhisperSource> {
    // Kalosm implements FromStr for WhisperSource; accept user strings like "quantized-distil-large-v3" etc.
    s.parse::<WhisperSource>()
        .map_err(|e| anyhow::anyhow!("invalid model '{s}': {e}"))
}

fn parse_language_opt(lang_opt: Option<&str>) -> Result<Option<WhisperLanguage>> {
    let Some(lang) = lang_opt.filter(|s| !s.eq_ignore_ascii_case("auto") && !s.trim().is_empty())
    else {
        return Ok(None);
    };
    lang.parse::<WhisperLanguage>()
        .map(Some)
        .map_err(|e| anyhow::anyhow!("invalid language '{lang}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;


}
