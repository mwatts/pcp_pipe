# Python Interop Removal & Native Mojo Enablement Plan

Goal: Transition the podcast processing pipeline from Python-dependent orchestration to a predominantly native Mojo implementation delivering true performance leverage (parallelism, SIMD, memory efficiency), while preserving correctness and gradually shrinking the Python surface until it becomes optional or removable.

---
## Guiding Principles
- Incremental replacement (avoid big bang rewrite)
- Maintain a continuously runnable pipeline (feature flags / adapter abstraction)
- Benchmark before & after each substitution (prove value)
- Prefer well-defined intermediate formats (JSONL, segments) to decouple stages
- Keep Python as *plugin* until fully displaced
- Push complexity to edges (download / model packaging) while hardening core pipeline

---
## Current Python Dependency Surface (Inventory)
| Capability | Python Lib | Used For | Replacement Strategy |
|------------|-----------|----------|----------------------|
| Downloading | yt-dlp | URL/media resolution + muxing | Phase 1 keep; Phase 3: optional native wrapper invoking yt-dlp via subprocess or pure HTTP fetch for simple MP3 URLs |
| Transcription | whisper (Py) | Model load + inference | Phase 2: Evaluate native Whisper (C++) binding or Mojo reimplementation of inference core; interim: isolate behind `Transcriber` trait |
| Summarization | transformers (DistilBART) | Abstractive summarization | Phase 4: Switch to extractive (Mojo-native) first; then optionally integrate a distilled encoder-decoder via ONNX runtime (if Mojo FFI available) |
| NER | spaCy | Entity extraction | Phase 4: Replace with rule + statistical lightweight model (ONNX) or curated gazetteers + chunk classification |
| Audio IO | librosa, numpy | Load waveform + duration | Phase 1: Replace with ffmpeg CLI (duration, conversion) then native WAV/MP3 decoder (Mojo) |
| Hashing | hashlib | URL hash | Replace with native hashing util (Phase 1) |
| JSON serialization | Python json | Results persistence | Mojo JSON or minimal writer (Phase 1) |
| Data containers | Python dict/list | Mixed-type storage | Define Mojo struct hierarchy + typed union / variant enums |

---
## Phased Roadmap Overview
| Phase | Focus | Python Reduction % (est.) | Exit Criteria |
|-------|-------|---------------------------|---------------|
| 0 | Baseline Hardening | 0% | Benchmark & logging established |
| 1 | Core Data & IO Native | 20–30% | Native JSON, hashing, path, result structs; audio duration via ffprobe |
| 2 | Transcription Abstraction | 30–45% | `Transcriber` trait + adapter; single model load caching; preparation for native Whisper |
| 3 | Audio Pipeline Native | 50–60% | Native decode (WAV) + chunk streaming + memory pipeline |
| 4 | Summarization Refactor | 65–75% | Mojo extractive summarizer baseline (keywords, topic sentences) |
| 5 | NER Replacement | 80–90% | Lightweight NER (regex + statistical classifier) |
| 6 | Advanced ML Port | 90–100% | Native/FFI Whisper + optional ONNX summarizer |

---
## Phase 0: Baseline Hardening (Pre-Replacement)
Deliverables:
- Introduce `logging.mojo` with leveled logger (INFO/DEBUG/WARN/ERROR, `--log-level`)
- Add `benchmark.mojo` measuring: download, transcribe, summarize, entities, total
- Add `env_report.mojo` printing environment & GPU capabilities
- Wrap current pipeline with `PipelineStats` struct (timings & counters)
Acceptance:
- `./build/mojo_podcast_processor --env` prints environment
- Benchmark JSON output persisted to `./benchmarks/latest.json`

---
## Phase 1: Core Data & IO Native
Objectives:
- Remove Python dict/list usage for final result aggregation
- Replace hashing (md5) with native function (e.g., FNV-1a or xxHash binding) truncated to 12 chars
- Implement minimal JSON serializer (support: String, Float64, Bool, Array, Object) → `json_writer.mojo`
- Introduce strong types: `TranscriptSegment`, `Entity`, `ProcessingResult`
- Audio duration via `ffprobe` subprocess (bridge) or basic WAV header parser (target both: simple first)
Refactors:
- `MojoPodcastProcessor` returns `ProcessingResult` instead of PythonObject
- Provide Python bridge adapter that converts `ProcessingResult` for legacy compatibility flag `--legacy-json`
Acceptance:
- Pipeline runs with native JSON output by default
- Removing Python `json` import does not affect functionality
- No broad `except:` blocks remain in newly refactored modules

---
## Phase 2: Transcription Abstraction
Objectives:
- Define `trait Transcriber { fn transcribe(path: String) -> Transcript }`
- Provide `PythonWhisperTranscriber` adapter containing all remaining Python dependencies for transcription
- Add model cache (singleton or lazy static) to prevent reload across invocations in same process
- Stream segmentation: number of segments + timings captured early
- Prepare spec for native Whisper port (model weights layout, mel spectrogram pipeline separation)
Technical Notes:
- Extract mel-spectrogram computation blueprint: implement in Mojo first (still call Python for decoding + inference)
Acceptance:
- Main pipeline depends only on trait; Python code confined to a single file
- Benchmark shows model load time excluded after first run (cache hit)

---
## Phase 3: Audio Pipeline Native
Objectives:
- Native WAV decoder (PCM 16-bit & 32-bit float) supporting large file streaming in blocks
- Implement mel-spectrogram in Mojo (reuse Phase 2 spec)
- Optional: Use ffmpeg for non-WAV conversion (external) then process WAV natively
- Replace librosa dependency fully
Performance Targets:
- Memory footprint reduced (no full waveform Python array duplication)
- Spectrogram generation throughput meets or exceeds librosa baseline by ≥1.2x
Acceptance:
- Run pipeline on WAV input with `--no-python-audio` flag removing librosa import path

---
## Phase 4: Summarization Refactor (Extractive First)
Objectives:
- Implement extractive summarizer: TF-IDF + sentence ranking (similar to TextRank-lite)
- Provide pluggable summarizer trait `Summarizer`
- Keep Python DistilBART under `PythonAbstractiveSummarizer` adapter (optional)
- Provide heuristics for chunk summarization & stitching
Acceptance:
- Default summarizer uses Mojo extractive method
- Word count reduction ratio configurable
- Summaries pass basic coherence checks (non-empty, reduced length)

---
## Phase 5: NER Replacement
Objectives:
- Implement hybrid NER:
  - Fast regex/pattern detection (ORG, PERSON, DATE, MONEY, PERCENT, URL)
  - Gazetteer matching (custom dictionary file support)
  - Optional statistical CRF / averaged perceptron (small model serialized compactly)
- Provide `EntityExtractor` trait; existing spaCy path moved to adapter
- Deduplicate + merge overlapping entities natively (already partial logic in accelerators)
Acceptance:
- Accuracy baseline evaluation vs spaCy on sample set (≥60% F1 initial acceptable; document gap)
- Entities struct includes origin `method: rule|model|hybrid`

---
## Phase 6: Advanced ML Port
Objectives:
- Integrate native or FFI-based Whisper (options: direct port of encoder/decoder or ONNX runtime binding)
- Replace Python summarizer with ONNX inference path if abstractive retained
- Evaluate GPU kernels for mel + attention (SIMD + parallel loops)
- Provide end-to-end Python-free build path (`--pure-mojo`)
Acceptance:
- `--pure-mojo` completes pipeline (download still may shell out to yt-dlp/ffmpeg only)
- Performance benchmark shows ≥ previous Python-backed throughput

---
## Cross-Cutting Concerns
### Error Model
Introduce `enum PipelineError { Download(url), Transcription(detail), Summarization(detail), Entities(detail), IO(path), Internal(msg) }` with Display formatting. Replace silent catches.

### Logging
Add JSON logging mode `--log-format json` for machine ingestion; includes phase start/stop, durations, resource stats.

### Progress Events
Emit structured progress callbacks (download_start, download_complete, transcribe_chunk, summarize_done, save_complete) enabling future UI/API integration.

### Configuration
Central `Config` struct loaded from CLI + optional `pcp.toml` with precedence: CLI > env > file > defaults.

### Benchmarks & Regression Tracking
Benchmark dimensions: total_time, transcription_time, summary_time, entity_time, rss (resource) memory_peak, model_load_time, throughput (sec audio / sec wall), CPU%, GPU util (if queryable). Write JSON + markdown table.

---
## Refactoring Topology (Target Module Layout)
```
src/
  config.mojo
  logging.mojo
  types/
    transcript.mojo
    entities.mojo
    result.mojo
  io/
    download.mojo
    audio_decode.mojo
    json_writer.mojo
  ml/
    transcriber_trait.mojo
    python_whisper_adapter.mojo
    mel_spectrogram.mojo
    summarizer_trait.mojo
    extractive_summarizer.mojo
    python_abstractive_summarizer.mojo
    entity_extractor_trait.mojo
    rule_ner.mojo
    python_spacy_adapter.mojo
  pipeline/
    orchestrator.mojo
    stages.mojo
  util/
    hashing.mojo
    time_tracker.mojo
    progress_events.mojo
cli/
  main.mojo
bench/
  benchmark.mojo
```
(Initial restructure can be logical inside existing file, then physical split.)

---
## Migration Mechanics
| Step | Action | Risk Mitigation |
|------|--------|-----------------|
| 1 | Introduce traits & adapters (transcriber, summarizer, entity extractor) | Keep Python adapters default until stable |
| 2 | Replace result assembly with Mojo structs + JSON writer | Compare diff of legacy vs new JSON via test harness |
| 3 | Native hashing + remove hashlib import | Keep test verifying stable 12-char id for identical URL |
| 4 | Mel spectrogram Mojo implementation | Cross-check numeric output tolerance (MSE threshold) vs Python reference |
| 5 | Extractive summarizer introduction | Dual-run mode to compare lengths & basic ROUGE-L placeholder |
| 6 | Rule-based NER baseline | Evaluate precision/recall on sample; iterate |
| 7 | Remove Python audio decode path for WAV inputs | Fallback flag `--python-audio` retained until stable |
| 8 | Whisper native/FFI integration | Start with CPU path; add GPU later; gating flag `--native-whisper` |

---
## Testing Strategy
- Golden sample directory: `testdata/sample_short.wav`, known transcript snippet
- Snapshot tests for JSON (ignoring time/version fields)
- Numeric tolerance tests for mel spectrogram vs reference (abs diff < 1e-5 average)
- Performance regression guard: fail benchmark job if total_time > baseline * 1.25

---
## Tooling & Automation
- Add `scripts/gen_benchmark_report.py` (temporary Python ok) to render markdown tables (removed later)
- GitHub Actions (future): build + run minimal audio test pipeline headless
- Add `make` (or `just`) targets: `build`, `bench`, `test`, `pure` (pure mojo build attempt)

---
## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Whisper native port complexity | Timeline slip | Stage via mel + encoder partial; start with decode wrappers |
| Performance regressions | Undermine rationale | Continuous benchmarks & threshold alerts |
| Feature parity drift (NER quality) | User dissatisfaction | Document limitations, incremental improvement roadmap |
| Over-fragmentation early | Slow velocity | Delay physical file split until traits stable |
| JSON writer bugs | Corrupt output | Fuzz small random objects; snapshot test |

---
## Acceptance Metrics (End State)
- Pure Mojo path (except external tools) executable with `--pure-mojo` on WAV input
- ≥80% lines of logic Mojo (measured by excluding adapters directory)
- Performance: total wall time improvement ≥15% vs original Python-integrated baseline on benchmark clip
- Memory usage reduction ≥20% (no duplicate Python arrays)
- Cold start model load eliminated in warm runs (cache)

---
## Immediate Next Sprint (Actionable Backlog)
1. Introduce `types` (ProcessingResult, Entity, TranscriptSegment) & native JSON writer
2. Add hashing util; remove hashlib in pipeline
3. Wrap transcription in `Transcriber` trait (Python adapter)
4. Introduce logging module & replace prints (keep CLI summary prints)
5. Add benchmark harness capturing stage timings
(Then review before proceeding to mel spectrogram implementation.)

---
## Review Checklist for This Plan
- Are phase boundaries clear & value-based?
- Any missing dependencies or hidden Python calls?
- Is extractive summarization acceptable interim vs abstractive?
- Priority of NER vs summarization correct for user goals?

Provide feedback; upon approval I will begin Immediate Sprint tasks and open structured follow-up docs (arc42 alignment afterwards).
