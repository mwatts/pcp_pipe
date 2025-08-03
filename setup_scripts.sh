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
    if ! python3 --version | grep -q "Python 3\.[89]\|Python 3\.1[0-9]"; then
        print_error "Python 3.8+ required. Please install Python 3.8 or higher."
        exit 1
    fi
    print_status "Python version check passed"
    
    # Check for CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected, will use CUDA acceleration"
        export USE_CUDA=1
    else
        print_warning "No NVIDIA GPU detected, will use CPU-only mode"
        export USE_CUDA=0
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Large models may fail."
        else
            print_status "Memory check passed ($MEMORY_GB GB available)"
        fi
    fi
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
        source $HOME/.cargo/env
        
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

# Setup Mojo if available
setup_mojo() {
    print_step "Setting up Mojo for high-performance computing..."
    
    # Check if Mojo is already installed
    if command -v mojo &> /dev/null; then
        print_status "Mojo already installed"
        mojo --version
        export MOJO_AVAILABLE=1
        return
    fi
    
    # Check if MAX platform is available
    if command -v max &> /dev/null; then
        print_status "MAX platform detected, Mojo should be available"
        export MOJO_AVAILABLE=1
        return
    fi
    
    print_warning "Mojo not detected. Installing Modular MAX platform..."
    
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    case "$OS" in
        linux)
            if [ "$ARCH" = "x86_64" ]; then
                print_status "Installing MAX for Linux x86_64..."
                curl -s https://get.modular.com | sh -
                modular auth
                modular install max
            else
                print_warning "MAX not available for $OS $ARCH. Mojo acceleration disabled."
                export MOJO_AVAILABLE=0
                return
            fi
            ;;
        darwin)
            if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "x86_64" ]; then
                print_status "Installing MAX for macOS..."
                curl -s https://get.modular.com | sh -
                modular auth
                modular install max
            else
                print_warning "MAX not available for $OS $ARCH. Mojo acceleration disabled."
                export MOJO_AVAILABLE=0
                return
            fi
            ;;
        *)
            print_warning "MAX not available for $OS. Mojo acceleration disabled."
            export MOJO_AVAILABLE=0
            return
            ;;
    esac
    
    # Verify installation
    if command -v mojo &> /dev/null; then
        print_status "Mojo installed successfully"
        mojo --version
        export MOJO_AVAILABLE=1
    else
        print_warning "Mojo installation may have failed. Continuing without Mojo acceleration."
        export MOJO_AVAILABLE=0
    fi
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ "$USE_UV" = "1" ]; then
        print_status "Using uv for fast package installation..."
        
        # Core dependencies - all local models
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        uv pip install openai-whisper
        uv pip install transformers[torch]
        uv pip install resemblyzer
        uv pip install scikit-learn
        uv pip install spacy
        uv pip install librosa
        uv pip install soundfile
        uv pip install yt-dlp
        uv pip install requests
        uv pip install numpy
        uv pip install scipy
        uv pip install asyncio-mqtt
        
        # Install spaCy language model
        print_status "Installing spaCy language model..."
        uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
        
    else
        print_status "Using pip for package installation..."
        
        # Create requirements.txt for batch installation - all local models
        cat > requirements.txt << EOF
torch
torchvision 
torchaudio
openai-whisper
transformers[torch]
resemblyzer
scikit-learn
spacy
librosa
soundfile
yt-dlp
requests
numpy
scipy
asyncio-mqtt
EOF
        
        pip install -r requirements.txt
        python -m spacy download en_core_web_lg
        rm requirements.txt
    fi
    
    print_status "Python dependencies installed successfully"
}

# Setup project structure
setup_project() {
    print_step "Setting up project structure..."
    
    # Create directory structure
    mkdir -p podcast_processor/{mojo_accelerators,tests,output,models}
    
    # Copy main files
    if [ -f "podcast_processor.py" ]; then
        cp podcast_processor.py podcast_processor/
    fi
    
    if [ -f "mojo_accelerators.mojo" ] && [ "$MOJO_AVAILABLE" = "1" ]; then
        cp mojo_accelerators.mojo podcast_processor/mojo_accelerators/
        
        # Compile Mojo modules
        print_status "Compiling Mojo accelerators..."
        cd podcast_processor/mojo_accelerators
        mojo build mojo_accelerators.mojo -o accelerators
        cd ../..
    fi
    
    # Create configuration file
    cat > podcast_processor/config.json << EOF
{
    "use_mojo_acceleration": $MOJO_AVAILABLE,
    "use_cuda": $USE_CUDA,
    "default_output_dir": "./output",
    "default_whisper_model": "large-v3",
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
    },
    "performance_optimizations": {
        "async_processing": true,
        "parallel_summarization": true,
        "batch_entity_extraction": true,
        "simd_audio_processing": $MOJO_AVAILABLE
    }
}
EOF
    
    print_status "Project structure created"
}

# Performance validation
validate_setup() {
    print_step "Validating installation..."
    
    # Test Python imports
    python3 -c "
import torch
import whisper
import transformers
import spacy
import librosa
import yt_dlp
print('✓ All Python dependencies imported successfully')
"
    
    # Test CUDA if available
    if [ "$USE_CUDA" = "1" ]; then
        python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"
    fi
    
    # Test Mojo if available
    if [ "$MOJO_AVAILABLE" = "1" ]; then
        print_status "Testing Mojo acceleration..."
        if [ -f "podcast_processor/mojo_accelerators/accelerators" ]; then
            ./podcast_processor/mojo_accelerators/accelerators
        else
            print_warning "Mojo accelerators not compiled, testing basic Mojo..."
            echo 'print("Mojo test successful")' | mojo
        fi
    fi
    
    print_status "Validation complete"
}

# Create launch script
create_launcher() {
    print_step "Creating launcher script..."
    
    cat > podcast_processor.sh << 'EOF'
#!/bin/bash
# High-Performance Podcast Processor Launcher

# Activate environment if using virtual env
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# Add Mojo to path if available
if [ -d "$HOME/.modular" ]; then
    export PATH="$HOME/.modular/pkg/packages.modular.com_max/bin:$PATH"
fi

# Run the processor - fully local, no external tokens required
python3 podcast_processor/podcast_processor.py "$@"
EOF
    
    chmod +x podcast_processor.sh
    
    print_status "Launcher script created: ./podcast_processor.sh"
}

# Main installation flow
main() {
    print_step "Starting high-performance podcast processor setup..."
    
    check_system
    install_uv
    setup_mojo
    install_dependencies
    setup_project
    validate_setup
    create_launcher
    
    echo ""
    print_status "🎉 Installation complete!"
    echo ""
    echo "Usage examples:"
    echo "  Basic usage:    ./podcast_processor.sh 'https://example.com/podcast.mp3'"
    echo "  With GPU:       ./podcast_processor.sh 'url' --use-gpu"
    echo "  With Mojo:      ./podcast_processor.sh 'url' --use-mojo"
    echo "  Setup env:      ./podcast_processor.sh --setup-env"
    echo "  🔥 All processing runs locally - no external API tokens required!"
    echo ""
    
    if [ "$MOJO_AVAILABLE" = "1" ]; then
        print_status "🔥 Mojo acceleration enabled for maximum performance!"
    fi
    
    if [ "$USE_CUDA" = "1" ]; then
        print_status "⚡ CUDA acceleration available"
    fi
    
    echo "Configuration saved to: podcast_processor/config.json"
    echo "Output will be saved to: podcast_processor/output/"
}

# Run main function
main "$@"
