# PCP Pipe - High-Performance Podcast Processing Pipeline

A blazing-fast, GPU-accelerated podcast processing pipeline built in **Rust** that downloads, transcribes, analyzes, and summarizes podcast audio using state-of-the-art ML models - all running locally without external API dependencies.

## Features

- 🎙️ **Advanced Transcription**: OpenAI Whisper large-v3 with word-level timestamps
- 🏷️ **Entity Extraction**: Named entity recognition with spaCy
- 📝 **Smart Summarization**: Content summaries using DistilBART
- 🚀 **Multi-Platform GPU Acceleration**: CUDA, Metal (Apple Silicon), ROCm support
- 🔒 **Privacy-First**: All processing happens locally - no external APIs
- 🌐 **Universal Download**: Supports YouTube, direct URLs, and more via yt-dlp
- 🎯 **Type Safety**: Compile-time error checking prevents runtime issues



## Installation


### Command Line Options

- `--output-dir <dir>`: Output directory (default: `./podcast_output`)
- `--whisper-model <model>`: Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `--no-gpu`: Disable GPU acceleration
- `--help`: Show help message

### Supported URLs

- YouTube videos and playlists
- Direct audio file URLs (MP3, WAV, M4A, etc.)
- Podcast RSS feeds
- Most audio streaming platforms

## Output Format

The processor creates organized output in the specified directory:

```
podcast_output/
├── podcast_abc123_results.json    # Complete results with all data
└── podcast_abc123.wav             # Downloaded audio file
```

### JSON Output Structure

```json
{
  "source_url": "https://example.com/podcast.mp3",
  "audio_file_path": "./podcast_output/podcast_abc123.wav",
  "transcript": "Full transcript with timestamps...",
  "summary": "AI-generated summary of key points...",
  "entities": [
    {
      "text": "OpenAI",
      "label": "ORG",
      "start": 156,
      "end": 162
    }
  ],
  "processing_time": 45.2,
}
```

## Performance

## Models Used

All models run locally for privacy:
- **Whisper large-v3**: State-of-the-art speech recognition
- **spaCy en_core_web_sm**: Named entity recognition
- **DistilBART CNN**: Efficient text summarization
- **yt-dlp**: Universal audio/video downloading

## Architecture


## Development

