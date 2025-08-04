# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a High-Performance Podcast Processing Pipeline that downloads, transcribes, analyzes, and summarizes podcast audio. The pipeline uses local ML models (Whisper, Resemblyzer, spaCy, DistilBART) and supports GPU acceleration across different platforms (CUDA, Metal, ROCm).

## Key Commands

### Setup and Installation
```bash
# Initial setup (detects platform, installs dependencies, configures environment)
./setup_scripts.sh

# Verify environment setup
./podcast_processor.sh --setup-env
```

### Running the Processor
```bash
# Basic usage
./podcast_processor.sh 'https://example.com/podcast.mp3'

# With GPU acceleration (auto-detects available GPU)
./podcast_processor.sh 'url' --use-gpu

# With Mojo acceleration (if installed)
./podcast_processor.sh 'url' --use-mojo
```

### Development
```bash
# Install dependencies manually (if needed)
uv sync

# Run the Python script directly
python main.py 'url'
```

## Architecture

### Main Components
- **main.py**: Core processing pipeline with HighPerformancePodcastProcessor class
  - Async audio downloading (yt-dlp)
  - Whisper transcription with word-level timestamps
  - Speaker diarization using Resemblyzer embeddings
  - Entity extraction with spaCy
  - Summarization using DistilBART
  - Platform-specific GPU optimizations

### Key Design Patterns
1. **Async Processing**: Uses asyncio for concurrent operations (downloading, processing)
2. **Platform Detection**: Automatically detects and optimizes for available hardware (Metal on macOS ARM, CUDA on NVIDIA, etc.)
3. **Local-Only**: All models run locally without external API dependencies
4. **Modular Acceleration**: Optional Mojo integration for performance-critical sections

### Output Structure
Processes save results to `output_{timestamp}/` containing:
- `transcript.json`: Full transcript with timestamps and speaker labels
- `summary.txt`: Generated summary
- `entities.json`: Extracted named entities
- `metadata.json`: Processing details and configuration

## Important Implementation Details

1. **GPU Memory Management**: The processor implements automatic GPU memory management with fallback to CPU when needed
2. **Model Loading**: Models are downloaded automatically on first use via Hugging Face transformers
3. **Audio Processing**: Handles various audio formats through librosa and soundfile
4. **Speaker Diarization**: Uses spectral clustering on Resemblyzer embeddings for speaker separation

## Development Notes

- Python 3.8+ required (project configured for 3.13+ but code supports 3.8+)
- No test suite currently implemented
- No linting configuration - follow existing code style
- The main.py file is the primary entry point containing the full implementation
- Use setup_scripts.sh for initial setup as it configures platform-specific optimizations