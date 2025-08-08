//! Shared data types for PCP Pipe

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub source_url: String,
    pub audio_file_path: String,
    pub transcript: String,
    pub summary: String,
    pub processing_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let r = ProcessingResult {
            source_url: "https://example.com".into(),
            audio_file_path: "./podcast_output/foo.mp3".into(),
            transcript: "hello".into(),
            summary: "hi".into(),
            processing_time: 1.23,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: ProcessingResult = serde_json::from_str(&s).unwrap();
        assert_eq!(back.source_url, r.source_url);
    }
}
