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
