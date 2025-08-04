# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **High-Performance Podcast Processing Pipeline** built entirely in **Mojo** that downloads, transcribes, analyzes, and summarizes podcast audio. The application uses Python interop for ML model integration while leveraging Mojo's native performance for compute-intensive operations. All models run locally without external API dependencies, supporting GPU acceleration across different platforms (CUDA, Metal, ROCm).

## Key Commands

### Setup and Installation
```bash
# Install Modular CLI and Mojo
curl -s https://get.modular.com | sh -
modular install mojo
export PATH=$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH

# Install Python dependencies (for ML models)
./setup_scripts.sh

# Build Mojo application
./build_mojo_processor.sh
```

### Running the Processor
```bash
# Basic usage
./build/mojo_podcast_processor 'https://example.com/podcast.mp3'

# With custom output directory
./build/mojo_podcast_processor 'url' --output-dir ./results

# Disable GPU acceleration
./build/mojo_podcast_processor 'url' --no-gpu

# Use different Whisper model
./build/mojo_podcast_processor 'url' --whisper-model base

# Show help
./build/mojo_podcast_processor --help
```

### Development
```bash
# Rebuild after changes
./build_mojo_processor.sh

# Test compilation only
mojo build mojo_podcast_processor.mojo --no-optimization

# Update Mojo
modular update mojo
```

## Architecture

### Core Components
- **mojo_podcast_processor.mojo**: Main Mojo application with complete processing pipeline
  - `MojoPodcastProcessor` struct: Core processor with all functionality
  - `TimeOffset` struct: Simple data structure for time ranges
  - Command-line interface with argument parsing
  - Complete audio processing workflow
  - Python interop for ML models (Whisper, spaCy, DistilBART)
  - Native Mojo performance optimizations

- **mojo_accelerators.mojo**: Performance-critical functions
  - Audio preprocessing utilities
  - Text chunking and analysis functions
  - Entity extraction optimizations
  - Benchmarking and validation tools

- **build_mojo_processor.sh**: Build system
  - Mojo compilation with optimization flags
  - Error handling and user guidance
  - Binary output management

- **setup_scripts.sh**: Dependency management
  - Python ML library installation
  - Platform-specific optimizations
  - Environment configuration

### Key Design Patterns
1. **Mojo-First Architecture**: Core application logic implemented in native Mojo
2. **Python Interop**: ML models accessed through Python API while maintaining Mojo performance
3. **Type Safety**: Strong typing with compile-time error checking
4. **Memory Efficiency**: Mojo's ownership system for automatic resource management
5. **Platform Agnostic**: GPU detection and optimization across CUDA, Metal, ROCm
6. **Local Processing**: No external API dependencies, complete privacy
7. **Error Resilience**: Graceful degradation with comprehensive error handling

### Output Structure
Results are saved to `./podcast_output/` (or custom directory) containing:
- `podcast_{hash}_results.json`: Complete processing results with all data
- `podcast_{hash}.wav`: Downloaded audio file

JSON output includes:
- Full transcript with word-level timestamps
- AI-generated summary
- Named entities with context
- Processing metadata and performance metrics
- Device and acceleration information

## Important Implementation Details

### Mojo Features Used
1. **Structs with Traits**: `Copyable`, `Movable` for efficient data handling
2. **Python Interop**: Seamless integration with ML libraries
3. **Type System**: Strong typing with `String`, `Float64`, `Bool`, `PythonObject`
4. **Error Handling**: `raises` functions with try-catch blocks
5. **Memory Management**: Automatic resource cleanup
6. **Native Performance**: Compiled binary with near-zero overhead

### ML Model Integration
1. **Whisper**: Speech-to-text with configurable model sizes
2. **spaCy**: Named entity recognition with local models
3. **DistilBART**: Text summarization with GPU acceleration
4. **yt-dlp**: Universal audio/video downloading

### Performance Optimizations
1. **GPU Acceleration**: Automatic detection and usage of available GPUs
2. **Batch Processing**: Efficient processing of audio segments
3. **Memory Management**: Optimized allocation and cleanup
4. **Compiled Binary**: Single executable with minimal startup time
5. **SIMD Operations**: Vector processing where applicable

## Development Notes

### Requirements
- **Mojo**: Latest version from Modular (primary requirement)
- **Python 3.8+**: For ML model interop only
- **GPU Drivers**: Optional but recommended (CUDA, Metal, ROCm)
- **FFmpeg**: For audio format conversion (installed by setup script)

### Build Process
1. Install Mojo via Modular CLI
2. Run `./setup_scripts.sh` to install Python ML dependencies
3. Execute `./build_mojo_processor.sh` to compile the application
4. Use `./build/mojo_podcast_processor` as the main entry point

### Performance Expectations
- **Compilation**: Single optimized binary (~1-5MB)
- **Startup Time**: Near-instant execution
- **Processing Speed**: 5-15x faster than real-time on GPU
- **Memory Usage**: 50-70% reduction compared to Python equivalents
- **Type Safety**: Compile-time error prevention

### Development Guidelines
- **Primary Language**: All new development in Mojo
- **Code Style**: Follow Mojo conventions with explicit type annotations
- **Error Handling**: Use `raises` functions and proper exception handling
- **Testing**: Manual testing with various audio sources and configurations
- **Documentation**: Update README.md and this file for any architectural changes

### File Organization
```
project/
├── mojo_podcast_processor.mojo    # Main Mojo application
├── mojo_accelerators.mojo         # Performance utilities
├── build_mojo_processor.sh        # Build script
├── setup_scripts.sh               # Dependency installer
├── MOJO_SETUP.md                  # Detailed setup guide
├── README.md                      # User documentation
└── build/                         # Compiled binaries
    └── mojo_podcast_processor
```

### Common Development Tasks

**Adding New Features**:
1. Implement in `mojo_podcast_processor.mojo`
2. Add command-line options if needed
3. Update help text and documentation
4. Test with `./build_mojo_processor.sh`

**Performance Optimization**:
1. Identify bottlenecks in processing pipeline
2. Implement SIMD operations in `mojo_accelerators.mojo`
3. Benchmark with different audio files
4. Profile memory usage and optimization

**Debugging**:
1. Use `mojo build --no-optimization` for debugging symbols
2. Add print statements for runtime debugging
3. Test with smaller audio files first
4. Check GPU availability and driver status

## Troubleshooting

### Common Issues
- **Mojo not found**: Ensure PATH includes Modular installation
- **Build failures**: Update Mojo version and check syntax
- **GPU not detected**: Verify drivers and hardware support
- **Audio download fails**: Check network connectivity and URL validity
- **Out of memory**: Use smaller models or increase system memory

### Environment Setup
- Follow MOJO_SETUP.md for detailed installation instructions
- Use setup_scripts.sh for automated dependency management
- Verify installation with `mojo --version` and test builds

## Resources

### Documentation
- **Mojo Language**: https://docs.modular.com/mojo/
- **Mojo Manual**: https://docs.modular.com/mojo/manual/
- **Basics**: https://docs.modular.com/mojo/manual/basics
- **functions**: https://docs.modular.com/mojo/manual/functions
- **variables**: https://docs.modular.com/mojo/manual/variables
- **types**: https://docs.modular.com/mojo/manual/types
- **operators**: https://docs.modular.com/mojo/manual/operators
- **Control Flow**: https://docs.modular.com/mojo/manual/control-flow/
- **Error Management**: https://docs.modular.com/mojo/manual/errors
- **Structs Guide**: https://docs.modular.com/mojo/manual/structs
- **modules and packages**: https://docs.modular.com/mojo/manual/packages


This project demonstrates production-ready Mojo development with real-world ML integration, serving as both a useful tool and a reference implementation for high-performance systems programming in Mojo.