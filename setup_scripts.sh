#!/bin/bash
# High-Performance Podcast Processor Setup
# Leverages uv for fast package management and Mojo for acceleration

set -e

echo "🔥 High-Performance Podcast Processor Setup"
echo "============================================"

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

    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
        print_error "Python 3.8+ required. Please install Python 3.8 or higher."
        exit 1
    fi
    
    if [ "$PYTHON_MINOR" -ge 13 ]; then
        print_warning "Python 3.13+ detected. Some dependencies don't support Python 3.13 yet."
        print_warning "Please use Python 3.8-3.12 for full compatibility."
        print_status "Attempting to find compatible Python version..."
        
        # Try to find Python 3.12 or lower
        for version in 3.12 3.11 3.10 3.9 3.8; do
            if command -v python$version &> /dev/null; then
                print_status "Found python$version, using it instead"
                PYTHON_CMD="python$version"
                break
            fi
        done
        
        if [ -z "$PYTHON_CMD" ]; then
            print_error "No compatible Python version (3.8-3.12) found."
            print_error "Please install Python 3.12 or lower. You can use:"
            print_error "  brew install python@3.12  # for macOS"
            print_error "  Or use pyenv to manage Python versions"
            exit 1
        fi
    else
        PYTHON_CMD="python3"
        print_status "Python version $PYTHON_VERSION is compatible"
    fi

    # Detect system architecture and GPU capabilities
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')

    case "$OS" in
        darwin)
            if [ "$ARCH" = "arm64" ]; then
                print_status "macOS ARM64 detected - will use Metal acceleration"
                export USE_METAL=1
                export USE_CUDA=0
                export GPU_TYPE="metal"
            elif [ "$ARCH" = "x86_64" ]; then
                print_status "macOS Intel detected - CPU-only mode"
                export USE_METAL=0
                export USE_CUDA=0
                export GPU_TYPE="cpu"
            fi
            ;;
        linux)
            if command -v nvidia-smi &> /dev/null; then
                print_status "NVIDIA GPU detected - will use CUDA acceleration"
                export USE_CUDA=1
                export USE_METAL=0
                export GPU_TYPE="cuda"
            elif command -v rocm-smi &> /dev/null; then
                print_status "AMD GPU detected - will use ROCm acceleration"
                export USE_ROCM=1
                export USE_CUDA=0
                export USE_METAL=0
                export GPU_TYPE="rocm"
            else
                print_status "No GPU detected - CPU-only mode"
                export USE_CUDA=0
                export USE_METAL=0
                export USE_ROCM=0
                export GPU_TYPE="cpu"
            fi
            ;;
        *)
            print_warning "Unknown OS: $OS - defaulting to CPU-only mode"
            export USE_CUDA=0
            export USE_METAL=0
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
    export PYTHON_CMD="$PYTHON_CMD"
}

# Install uv for fast package management
install_uv() {
    print_step "Installing uv (fast Python package manager)..."

    if command -v uv &> /dev/null; then
        print_status "uv already installed"
        uv --version
    else
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            print_status "uv installed successfully"
            uv --version
        else
            print_error "Failed to install uv. Falling back to pip."
            export USE_UV=0
            return
        fi
    fi
    export USE_UV=1
}

# Setup Mojo with proper macOS ARM support
setup_mojo() {
    print_step "Setting up Mojo for high-performance computing..."

    # Check if Mojo is already installed
    if command -v mojo &> /dev/null; then
        print_status "Mojo already installed"
        mojo --version
        export MOJO_AVAILABLE=1
        return
    fi

    print_status "Mojo installation available but optional. Skipping for now."
    print_status "To install Mojo later, visit: https://www.modular.com/max/install"
    export MOJO_AVAILABLE=0
}

# Create virtual environment
create_venv() {
    print_step "Creating virtual environment..."
    
    if [ -d ".venv" ]; then
        print_status "Virtual environment already exists"
    else
        $PYTHON_CMD -m venv .venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    print_status "Virtual environment activated"
}

# Install Python dependencies with correct PyTorch for platform
install_dependencies() {
    print_step "Installing Python dependencies for $SYSTEM_OS $SYSTEM_ARCH..."

    if [ "$USE_UV" = "1" ]; then
        print_status "Using uv for fast package installation..."

        # Install PyTorch with correct backend for the platform
        case "$GPU_TYPE" in
            metal)
                print_status "Installing PyTorch with Metal acceleration for macOS ARM..."
                uv pip install torch torchvision torchaudio
                ;;
            cuda)
                print_status "Installing PyTorch with CUDA acceleration..."
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                ;;
            rocm)
                print_status "Installing PyTorch with ROCm acceleration..."
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
                ;;
            *)
                print_status "Installing PyTorch CPU-only version..."
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                ;;
        esac

        # Install project with all dependencies
        print_status "Installing project dependencies..."
        uv pip install -e .

        # Install spaCy language model
        print_status "Installing spaCy language model..."
        uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

    else
        print_status "Using pip for package installation..."

        # Install PyTorch first with correct backend
        case "$GPU_TYPE" in
            metal)
                pip install torch torchvision torchaudio
                ;;
            cuda)
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                ;;
            rocm)
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
                ;;
            *)
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                ;;
        esac

        # Install project
        pip install -e .
        python -m spacy download en_core_web_lg
    fi

    print_status "Python dependencies installed successfully for $GPU_TYPE acceleration"
}

# Setup project structure with platform-specific optimizations
setup_project() {
    print_step "Setting up project structure..."

    # Create output directory
    mkdir -p podcast_output

    # Create platform-optimized configuration
    cat > config.json << EOF
{
    "system_info": {
        "os": "$SYSTEM_OS",
        "architecture": "$SYSTEM_ARCH",
        "gpu_type": "$GPU_TYPE"
    },
    "acceleration": {
        "use_mojo": $MOJO_AVAILABLE,
        "use_metal": ${USE_METAL:-0},
        "use_cuda": ${USE_CUDA:-0},
        "use_rocm": ${USE_ROCM:-0}
    },
    "model_config": {
        "default_whisper_model": "large-v3",
        "use_local_models_only": true,
        "speaker_diarization": "resemblyzer"
    },
    "performance": {
        "max_concurrent_downloads": 4,
        "batch_sizes": {
            "cpu": {
                "summarization": 1,
                "entity_extraction": 8
            },
            "gpu": {
                "summarization": 4,
                "entity_extraction": 32
            }
        }
    },
    "optimizations": {
        "async_processing": true,
        "parallel_summarization": true,
        "batch_entity_extraction": true,
        "simd_audio_processing": $MOJO_AVAILABLE,
        "local_only_processing": true
    }
}
EOF

    print_status "Project structure created with $GPU_TYPE optimizations"
}

# Performance validation with platform-specific tests
validate_setup() {
    print_step "Validating installation for $SYSTEM_OS $SYSTEM_ARCH..."

    # Test Python imports
    $PYTHON_CMD -c "
import torch
import whisper
import transformers
import spacy
import librosa
import yt_dlp
import resemblyzer
import sklearn
print('✓ All Python dependencies imported successfully')
"

    # Test GPU acceleration based on platform
    case "$GPU_TYPE" in
        metal)
            $PYTHON_CMD -c "
import torch
print('PyTorch version:', torch.__version__)
print('Metal Performance Shaders available:', torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('✓ Metal acceleration ready')
    # Test basic tensor operation
    x = torch.randn(100, 100, device=device)
    y = torch.mm(x, x.t())
    print('✓ Metal tensor operations working')
else:
    print('ℹ Metal not available, using CPU')
"
            ;;
        cuda)
            $PYTHON_CMD -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))
    print('✓ CUDA acceleration ready')
else:
    print('ℹ CUDA not available, using CPU')
"
            ;;
        rocm)
            $PYTHON_CMD -c "
import torch
print('ROCm available:', torch.cuda.is_available())  # PyTorch uses same API for ROCm
if torch.cuda.is_available():
    print('✓ ROCm acceleration ready')
else:
    print('ℹ ROCm not available, using CPU')
"
            ;;
        *)
            $PYTHON_CMD -c "
import torch
print('PyTorch CPU version:', torch.__version__)
print('✓ CPU-only mode configured')
"
            ;;
    esac

    # Test Resemblyzer for speaker diarization
    $PYTHON_CMD -c "
from resemblyzer import VoiceEncoder
import torch

device = 'cpu'
# Test appropriate device for platform
if torch.backends.mps.is_available():
    device = 'cpu'  # Resemblyzer doesn't support MPS yet
elif torch.cuda.is_available():
    device = 'cuda'

print(f'Testing Resemblyzer on {device}...')
try:
    encoder = VoiceEncoder(device=device)
    print('✓ Resemblyzer loaded successfully')
    print(f'✓ Speaker diarization ready on {device}')
except Exception as e:
    print(f'⚠ Resemblyzer warning: {e}')
    print('Will fallback to CPU if needed')
"

    print_status "✓ Platform validation complete for $SYSTEM_OS $SYSTEM_ARCH with $GPU_TYPE acceleration"
}

# Create platform-optimized launcher script
create_launcher() {
    print_step "Creating platform-optimized launcher script..."

    cat > podcast_processor.sh << 'EOF'
#!/bin/bash
# High-Performance Podcast Processor Launcher

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Platform-specific environment optimization
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    darwin)
        # macOS optimizations
        export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
        export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
        
        if [ "$ARCH" = "arm64" ]; then
            # Apple Silicon optimizations
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        fi
        ;;
    linux)
        # Linux optimizations
        export OMP_NUM_THREADS=$(nproc)
        export MKL_NUM_THREADS=$(nproc)
        export NUMEXPR_NUM_THREADS=$(nproc)
        ;;
esac

# Run the processor
python main.py "$@"
EOF

    chmod +x podcast_processor.sh

    print_status "Platform-optimized launcher created: ./podcast_processor.sh"
}

# Main installation flow
main() {
    print_step "Starting high-performance podcast processor setup..."

    check_system
    install_uv
    setup_mojo
    create_venv
    install_dependencies
    setup_project
    validate_setup
    create_launcher

    echo ""
    print_status "🎉 Installation complete for $SYSTEM_OS $SYSTEM_ARCH!"
    echo ""
    print_status "Platform Configuration:"
    echo "  OS/Architecture: $SYSTEM_OS/$SYSTEM_ARCH"
    echo "  GPU Type: $GPU_TYPE"
    echo "  Mojo Available: $([ "$MOJO_AVAILABLE" = "1" ] && echo "Yes" || echo "No")"
    echo ""
    echo "Usage examples:"
    echo "  Basic usage:    ./podcast_processor.sh 'https://example.com/podcast.mp3'"
    echo "  With GPU:       ./podcast_processor.sh 'url' --use-gpu"
    echo "  With Mojo:      ./podcast_processor.sh 'url' --use-mojo"
    echo "  Setup env:      ./podcast_processor.sh --setup-env"
    echo "  🔥 All processing runs locally - no external API tokens required!"
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

    echo "Configuration saved to: config.json"
    echo "Output will be saved to: podcast_output/"
}

# Run main function
main "$@"