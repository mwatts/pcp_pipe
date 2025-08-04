#!/bin/bash
# Build script for Mojo accelerators
# This script will compile Mojo modules when the Mojo compiler is available

set -e

echo "🔥 Building Mojo Accelerators for Podcast Processing Pipeline"
echo "============================================================="

# Check if Mojo is installed
if ! command -v mojo &> /dev/null; then
    echo "❌ Mojo compiler not found!"
    echo "Please install Modular CLI and Mojo:"
    echo "  1. Install Modular CLI: curl -s https://get.modular.com | sh -"
    echo "  2. Install Mojo: modular install mojo"
    echo "  3. Add to PATH: export PATH=\$HOME/.modular/pkg/packages.modular.com_mojo/bin:\$PATH"
    echo ""
    echo "For now, the pipeline will use Python fallback implementations."
    exit 1
fi

echo "✅ Mojo compiler found: $(mojo --version)"

# Check if source file exists
if [ ! -f "mojo_accelerators.mojo" ]; then
    echo "❌ mojo_accelerators.mojo not found!"
    exit 1
fi

echo "📝 Source file found: mojo_accelerators.mojo"

# Create build directory
mkdir -p build

echo "🔨 Compiling Mojo accelerators..."

# Compile Mojo to executable (for testing)
echo "  → Building standalone executable..."
mojo build mojo_accelerators.mojo -o build/mojo_accelerators

# Compile to Python package (when available)
echo "  → Building Python extension..."
echo "     Note: Python extension compilation via 'mojo package' when available"
echo "     This would create importable Python modules from Mojo code"
echo "     Current status: Feature in development by Modular team"

echo ""
echo "✅ Build complete!"
echo ""
echo "To test the Mojo accelerators:"
echo "  ./build/mojo_accelerators"
echo ""
echo "To enable in the podcast processor:"
echo "  1. Wait for Mojo Python packaging features"
echo "  2. Update mojo_interface.py to import compiled modules"
echo "  3. Set MOJO_AVAILABLE = True in the interface"
echo "  4. Or use Mojo as standalone executable with file I/O interface"