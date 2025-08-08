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

## 10. Quality Scenarios
- Fast processing, robust against network hiccups, reproducible builds.

## 11. Risks and Technical Debt
- YouTube without yt-dlp deferred; codec coverage via symphonia.

## 12. Glossary
- ASR: Automatic Speech Recognition; LLM: Large Language Model.
