# PCP Pipe Implementation Plan (Revised)

This plan reflects updated constraints: pure-Rust downloads (no yt-dlp), no ffmpeg/WAV conversion, remove entity extraction and Python entirely, and use kalosm to host/run LLMs locally in Rust.

Docs for kalosm: https://docs.rs/kalosm/latest/kalosm/index.html

## Requirements checklist (updated)
- [ ] CLI with options: --output-dir, --whisper-model [tiny|base|small|medium|large-v3], --no-gpu, --help
- [ ] URL inputs: YouTube (where feasible), direct audio URLs, podcast RSS feeds
- [ ] Download using Rust crates only (no yt-dlp). Handle redirects, playlists/feeds expansion.
- [ ] No ffmpeg usage; do not convert to WAV. Support decoding audio directly in Rust for transcription.
- [ ] Transcription with Whisper (large-v3 capable) with word-level timestamps
- [ ] Summarization using kalosm LLMs (local inference)
- [ ] Remove entity extraction (NER) and all Python dependencies
- [ ] Multi-platform acceleration: CUDA, Metal (Apple Silicon), ROCm where applicable (via underlying libs)
- [ ] Privacy-first: no external inference APIs
- [ ] Structured JSON output matching README schema (sans entities)
- [ ] Rust workspace layout: crates/ for libraries, bin/ for CLI, tests per crate
- [ ] Documentation under docs/ using arc42; diagrams in docs/diagrams (d2 preferred)

## High-level architecture
- Orchestrator (pipeline) coordinates stages and produces final JSON.
- Stages:
  1) Fetcher: input URL(s) -> audio bytes/stream + metadata
  2) Decoder: audio stream -> PCM frames (in-memory), with timing
  3) Transcriber: PCM -> transcript with word timestamps
  4) Summarizer: transcript -> abstractive summary via kalosm
- Device manager selects acceleration backend (CUDA/MPS/ROCm/CPU) where supported.
- Shared types crate ensures consistent data structures across stages.

## Technology choices (Rust-only)
- HTTP/Networking: reqwest (blocking or async via tokio), robust redirect handling, range requests, retries.
- RSS/Feeds: rss or feed-rs for parsing.
- YouTube support (without yt-dlp):
  - Attempt using the innertube API via crates like youtube_dl_rs alternatives or ytapi; if not viable, limit to direct audio URLs and RSS enclosures initially, add YouTube later as a feature.
  - Assumption for M0-M1: prioritize direct audio URLs and RSS enclosures; document YouTube as deferred.
- Audio decoding: symphonia for container/codec demux/decoding (MP3, M4A/AAC, WAV, FLAC, OGG). Avoid transcoding; decode to PCM in-memory for Whisper.
- Whisper: whisper-rs (whisper.cpp) for transcription with timestamps; feed PCM frames directly.
- Summarization: kalosm for local LLM hosting/inference; pick a small efficient model for summaries; chunk long transcripts.
- CLI: clap (derive), anyhow/thiserror for errors.
- Async runtime: tokio for IO+process mgmt; rayon optional for CPU-bound parallelism.

- Logging: tracing + tracing-subscriber.

## Data contracts
- Input: one or more URLs (string) on CLI.
- Output JSON fields:
  - source_url: string
  - audio_file_path: string (downloaded original container file path)
  - transcript: string
  - summary: string
  - processing_time: seconds (float)
  (Note: no entities field; intentionally omitted.)
- Filesystem outputs under output-dir per item using stable IDs. Store the original file as-is (no WAV conversion).

## Workspace layout
- crates/
  - types/         (shared structs, JSON schemas)
  - fetcher/       (reqwest-based downloader; RSS expansion; optional YouTube later)
  - decoder/       (symphonia-based decode -> PCM frames API)
  - transcribe/    (whisper-rs integration + model mgmt)
  - summarize/     (kalosm wrapper for summarization)
  - pipeline/      (orchestration, device selection, error boundaries)
  - utils/         (logging, config, paths, time)
- bin/
  - pcp-cli/       (CLI using clap; wires pipeline)
- docs/
  - architecture/  (arc42; references diagrams)
  - diagrams/      (*.d2 preferred, mermaid fallback)
- podcast_output/ (default output dir)

## Milestones and deliverables

M0: Scaffolding and docs (1–2 days)
- Cargo workspace with crates and CLI bin per conventions.
- docs/ with arc42 skeleton and initial d2 diagram stubs.
- GitHub Actions: fmt, clippy, build matrix (macOS, ubuntu), basic tests.

M1: Fetcher + Storage (1–2 days)
- Implement direct URL download with reqwest, robust file naming, checksum, resume (HTTP ranges if server supports).
- RSS support: parse feed, extract audio enclosures, enqueue per-episode.
- Defer YouTube support unless a reliable Rust-only approach is available; document as future work.
- Tests: unit tests for URL normalization and RSS parsing; integration test behind feature flag for network.

M2: Decoder (1–2 days)
- Integrate symphonia to decode supported formats to PCM f32/i16 frames.
- Provide iterator/streaming API with sample rate conversion if needed for Whisper (e.g., 16kHz mono).
- Tests: decode tiny included sample files covering MP3/M4A/WAV.

M3: Transcription (2–4 days)
- whisper-rs integration; model management (download gguf to cache if needed).
- Word-level timestamps; device selection (MPS/CUDA/CPU) and --no-gpu flag.
- Tests: smoke test on a few-second sample; allow skipping in CI.

M4: Summarization with kalosm (2–3 days)
- Wrap kalosm to load a compact model locally.
- Implement chunking for long transcripts and merge summaries.
- Tests: deterministic/snapshot testing on short text (acknowledging LLM variance; use prompts that minimize variation).

M5: Pipeline integration (2–3 days)
- Orchestrate fetch -> decode -> transcribe -> summarize.
- Compose final JSON (without entities) and persist per item.
- Progress logging and timing; partial failure handling.

M6: CLI polish + UX (1–2 days)
 - Implement CLI flags from README.
- Add --concurrency and --device [auto|cpu|gpu] options.

M7: Performance & caching (2–3 days)
- Cache models; reuse across items; tune thread pools.
- Streaming decode to transcription to reduce memory.

M8: Packaging and release (1–2 days)
- Release binaries for macOS (arm64/x86_64) and Linux.
- Versioning and CHANGELOG (Conventional Commits).

## Risks and mitigations
- YouTube without yt-dlp is non-trivial: mark as deferred or experimental; focus on direct URLs/RSS first.
- Codec coverage: rely on symphonia; document unsupported formats; provide helpful errors.
- LLM determinism for tests: use constrained prompts; snapshot with tolerance or skip in CI.
- GPU backend variance: detect at runtime; CPU fallback; diagnostics.

## Testing strategy
- Unit tests per crate; integration tests for pipeline and CLI.
- Include tiny audio samples for decode/transcribe smoke tests.
- Feature flags: offline/no-network for CI, cpu-only mode, small-models.
- JSON schema validation for outputs (entities removed).

## Documentation
- arc42 architecture in docs/architecture; update diagrams to remove NER and Python.
- Quickstart and troubleshooting guides; GPU setup notes (MPS/CUDA/ROCm where relevant).

## Next steps (M0 start)
1) Scaffold cargo workspace and initial crates (types, fetcher, decoder, transcribe, summarize, pipeline) and bin/pcp-cli.
2) Add docs/ skeleton with arc42 placeholders and initial d2 diagram stubs.
3) Implement Fetcher happy path (direct URL) and CLI that accepts --output-dir and URL, writing a skeleton JSON (no ffmpeg, store original file).
4) Land CI with fmt/clippy/build and a single fetcher unit test.
