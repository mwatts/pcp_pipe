# Architecture (arc42 Skeleton)

## 1. Introduction and Goals
- High-performance, local, privacy-first podcast processing pipeline in Rust.

## 2. Constraints
- Rust-only downloads and processing; no external API inference.
- No ffmpeg; in-Rust decode with symphonia.
- No Python/NER; summarization via kalosm.

## 3. Context and Scope
- Inputs: URLs and RSS feeds.
- Outputs: JSON results + downloaded originals.

## 4. Solution Strategy
- Crate-per-stage; async orchestration; device autodetect.

### Why a Decoder (M2) is required
- Whisper consumes log-Mel spectrograms computed from raw PCM audio (typically 16 kHz, mono, float32 in [-1, 1]).
- Compressed formats (MP3, AAC/M4A, OGG, etc.) are encoded bitstreams in containers and cannot be ingested directly by Whisper. They must be demuxed and decoded to PCM first.
- Sample rate and channels vary (e.g., 44.1/48 kHz, stereo). We must resample to 16 kHz and downmix to mono to match Whisper’s expected input and produce correct timings.
- The whisper.cpp/whisper-rs APIs accept PCM buffers (or WAV as a convenience) and do not parse arbitrary containers. Many examples rely on ffmpeg for decoding—disallowed per constraints—so we use a Rust decoder.
- Symphonia provides pure-Rust demux/codec support for common audio types; this satisfies our “Rust-only” constraint while enabling robust streaming, duration consistency (VBR aware), and proper handling of tags (ID3) and edge cases.

- Benefits of explicit decoding:
	- Enables streaming decode for long files (bounded memory).
	- Ensures stable duration and timestamp alignment for word-level ASR.
	- Normalizes audio (sample rate, channel layout) for consistent ASR accuracy.

## 5. Building Block View
- fetcher -> decoder -> transcribe -> summarize -> JSON

## 6. Runtime View
- Pipeline per URL, minimal shared state, cached models.

## 7. Deployment View
- Local binary, optional GPU acceleration.

## 8. Crosscutting Concepts
- tracing, error handling, config, model cache.

## 9. Architecture Decisions
- Use symphonia for decoding, whisper-rs for ASR, kalosm for LLM.
	- Rationale: symphonia is an actively maintained, well-documented, pure-Rust library with broad codec/container support and good performance; avoids ffmpeg.

## 10. Quality Scenarios
- Fast processing, robust against network hiccups, reproducible builds.

## 11. Risks and Technical Debt
- YouTube without yt-dlp deferred; codec coverage via symphonia.

## 12. Glossary
- ASR: Automatic Speech Recognition; LLM: Large Language Model.
