//! Transcription: placeholder for whisper-rs integration.

pub struct Transcript {
    pub text: String,
}

pub fn transcribe_pcm(_frames: &crate::Transcript) -> anyhow::Result<Transcript> {
    // Intentional placeholder; will change signature in M3 to accept PCM.
    Ok(Transcript { text: String::new() })
}
