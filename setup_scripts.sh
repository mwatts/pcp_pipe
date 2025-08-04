#!/bin/bash
# High-Performance Mojo Podcast Processor Setup
# Installs Mojo and Python ML dependencies for the podcast processing pipeline

set -e

echo "🎙️ Mojo Podcast Processor Setup"
echo "================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check system requirements
check_system() {
    print_step "Checking system requirements..."

    # Check Python version (needed for ML models)
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
        print_error "Python 3.8+ required for ML model support."
        exit 1
    fi
    
    print_status "Python version $PYTHON_VERSION is compatible"

    # Detect system architecture and GPU capabilities
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')

    case "$OS" in
        darwin)
            if [ "$ARCH" = "arm64" ]; then
                print_status "macOS ARM64 detected - Metal acceleration available"
                export GPU_TYPE="metal"
            else
                print_status "macOS Intel detected - CPU-only mode"
                export GPU_TYPE="cpu"
            fi
            ;;
        linux)
            if command -v nvidia-smi &> /dev/null; then
                print_status "NVIDIA GPU detected - CUDA acceleration available"
                export GPU_TYPE="cuda"
            elif command -v rocm-smi &> /dev/null; then
                print_status "AMD GPU detected - ROCm acceleration available"
                export GPU_TYPE="rocm"
            else
                print_status "No GPU detected - CPU-only mode"
                export GPU_TYPE="cpu"
            fi
            ;;
        *)
            print_warning "Unknown OS: $OS - defaulting to CPU-only mode"
            export GPU_TYPE="cpu"
            ;;
    esac

    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Large models may fail."
        else
            print_status "Memory check passed ($MEMORY_GB GB available)"
        fi
    elif command -v sysctl &> /dev/null; then
        # macOS memory check
        MEMORY_BYTES=$(sysctl -n hw.memsize)
        MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Large models may fail."
        else
            print_status "Memory check passed ($MEMORY_GB GB available)"
        fi
    fi

    export SYSTEM_ARCH="$ARCH"
    export SYSTEM_OS="$OS"
}

# Install Mojo
install_mojo() {
    print_step "Setting up Mojo..."

    # Check if Mojo is already installed
    if command -v mojo &> /dev/null; then
        print_status "Mojo already installed"
        mojo --version
        return
    fi

    print_status "Installing Modular CLI and Mojo..."
    print_status "This will download and install Mojo from Modular"
    
    # Install Modular CLI
    if ! command -v modular &> /dev/null; then
        print_status "Installing Modular CLI..."
        curl -s https://get.modular.com | sh -
        
        # Source the shell configuration to get modular in PATH
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
        if [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc" 2>/dev/null || true
        fi
        
        # Add to current session PATH
        export PATH="$HOME/.modular/bin:$PATH"
    fi

    # Install Mojo
    if command -v modular &> /dev/null; then
        print_status "Installing Mojo..."
        modular install mojo
        
        # Add Mojo to PATH
        export PATH="$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH"
        
        # Verify installation
        if command -v mojo &> /dev/null; then
            print_status "Mojo installed successfully"
            mojo --version
            
            # Add to shell profile for persistence
            SHELL_PROFILE=""
            if [ -n "$ZSH_VERSION" ]; then
                SHELL_PROFILE="$HOME/.zshrc"
            elif [ -n "$BASH_VERSION" ]; then
                SHELL_PROFILE="$HOME/.bashrc"
            fi
            
            if [ -n "$SHELL_PROFILE" ]; then
                echo 'export PATH="$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH"' >> "$SHELL_PROFILE"
                print_status "Added Mojo to PATH in $SHELL_PROFILE"
            fi
        else
            print_error "Mojo installation failed"
            exit 1
        fi
    else
        print_error "Failed to install Modular CLI"
        exit 1
    fi
}

# Install Python dependencies for ML models
install_python_dependencies() {
    print_step "Installing Python dependencies for ML models..."

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    print_status "Virtual environment activated"

    # Upgrade pip
    pip install --upgrade pip

    # Install PyTorch with correct backend for the platform
    case "$GPU_TYPE" in
        metal)
            print_status "Installing PyTorch with Metal acceleration for macOS ARM..."
            pip install torch torchvision torchaudio
            ;;
        cuda)
            print_status "Installing PyTorch with CUDA acceleration..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        rocm)
            print_status "Installing PyTorch with ROCm acceleration..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
            ;;
        *)
            print_status "Installing PyTorch CPU-only version..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac

    # Install ML libraries needed by Mojo application
    print_status "Installing ML model dependencies..."
    pip install openai-whisper transformers[torch] spacy librosa soundfile yt-dlp requests numpy scipy scikit-learn

    # Install spaCy language model
    print_status "Installing spaCy language model..."
    python -m spacy download en_core_web_sm
    
    # Try to install larger model if available
    python -m spacy download en_core_web_lg 2>/dev/null || print_warning "Large spaCy model not available, using small model"

    print_status "Python dependencies installed successfully for $GPU_TYPE acceleration"
}

# Build Mojo application
build_mojo_app() {
    print_step "Building Mojo podcast processor..."

    if ! command -v mojo &> /dev/null; then
        print_error "Mojo not found. Please ensure Mojo is installed and in PATH."
        exit 1
    fi

    # Create build directory
    mkdir -p build

    # Build the Mojo application
    print_status "Compiling Mojo application..."
    if ./build_mojo_processor.sh; then
        print_status "Mojo application built successfully"
    else
        print_error "Failed to build Mojo application"
        exit 1
    fi
}

# Validate installation
validate_setup() {
    print_step "Validating installation..."

    # Test Mojo
    if command -v mojo &> /dev/null; then
        print_status "✓ Mojo available: $(mojo --version)"
    else
        print_error "✗ Mojo not available"
        exit 1
    fi

    # Test Mojo application
    if [ -f "./build/mojo_podcast_processor" ]; then
        print_status "✓ Mojo podcast processor built"
    else
        print_error "✗ Mojo podcast processor not found"
        exit 1
    fi

    # Activate virtual environment for Python tests
    source .venv/bin/activate 2>/dev/null || true

    # Test Python imports
    python3 -c "
import torch
import whisper
import transformers
import spacy
import librosa
import yt_dlp
print('✓ All Python dependencies imported successfully')
" 2>/dev/null && print_status "✓ Python ML libraries available" || print_warning "Some Python libraries may have issues"

    # Test GPU acceleration based on platform
    case "$GPU_TYPE" in
        metal)
            python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✓ Metal acceleration ready')
else:
    print('ℹ Metal not available, using CPU')
" 2>/dev/null
            ;;
        cuda)
            python3 -c "
import torch
if torch.cuda.is_available():
    print('✓ CUDA acceleration ready')
else:
    print('ℹ CUDA not available, using CPU')
" 2>/dev/null
            ;;
        *)
            print_status "✓ CPU-only mode configured"
            ;;
    esac

    print_status "✓ Installation validation complete"
}

# Create project structure
setup_project() {
    print_step "Setting up project structure..."

    # Create output directory
    mkdir -p podcast_output

    print_status "Project structure created"
}

# Main installation flow
main() {
    print_step "Starting Mojo podcast processor setup..."

    check_system
    install_mojo
    install_python_dependencies
    setup_project
    build_mojo_app
    validate_setup

    echo ""
    print_status "🎉 Installation complete!"
    echo ""
    print_status "Platform Configuration:"
    echo "  OS/Architecture: $SYSTEM_OS/$SYSTEM_ARCH"
    echo "  GPU Type: $GPU_TYPE"
    echo "  Mojo: $(mojo --version 2>/dev/null || echo 'Not available')"
    echo ""
    echo "Usage:"
    echo "  ./build/mojo_podcast_processor 'https://example.com/podcast.mp3'"
    echo "  ./build/mojo_podcast_processor --help"
    echo ""
    echo "The processor runs entirely in Mojo with Python ML model integration."
    echo "All processing happens locally - no external API tokens required!"
    echo ""

    case "$GPU_TYPE" in
        metal)
            print_status "🍎 Metal acceleration configured for Apple Silicon"
            ;;
        cuda)
            print_status "⚡ CUDA acceleration available"
            ;;
        rocm)
            print_status "🔥 ROCm acceleration available"
            ;;
        *)
            print_status "🖥️ CPU-only mode configured"
            ;;
    esac

    echo ""
    echo "Next steps:"
    echo "1. Test the installation: ./build/mojo_podcast_processor --help"
    echo "2. Process your first podcast: ./build/mojo_podcast_processor 'your-url'"
    echo "3. Check the output in: ./podcast_output/"
}

# Run main function
main "$@"