#!/bin/bash

# Build script for Mojo Podcast Processor
# This script compiles the standalone Mojo podcast processor

set -e

echo "🎙️ Building Mojo Podcast Processor..."

# Check for Mojo compiler
if ! command -v mojo &> /dev/null; then
    echo "❌ Mojo compiler not found!"
    echo ""
    echo "To install Mojo:"
    echo "1. Install Modular CLI: curl -s https://get.modular.com | sh -"
    echo "2. Install Mojo: modular install mojo"
    echo "3. Add to PATH: export PATH=\$HOME/.modular/pkg/packages.modular.com_mojo/bin:\$PATH"
    echo "4. Verify: mojo --version"
    echo ""
    echo "For more details, see: MOJO_SETUP.md"
    exit 1
fi

# Create build directory
mkdir -p build

# Display Mojo version
echo "✓ Using Mojo compiler:"
mojo --version

# Compile the main processor
echo "🔨 Compiling mojo_podcast_processor.mojo..."
mojo build mojo_podcast_processor.mojo -o build/mojo_podcast_processor

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Executable created: build/mojo_podcast_processor"
    echo "Size: $(ls -lh build/mojo_podcast_processor | awk '{print $5}')"
    echo ""
    echo "Usage:"
    echo "  ./build/mojo_podcast_processor 'https://example.com/podcast.mp3'"
    echo "  ./build/mojo_podcast_processor --help"
    echo ""
    echo "🎉 Mojo podcast processor is ready to use!"
else
    echo "❌ Build failed!"
    exit 1
fi