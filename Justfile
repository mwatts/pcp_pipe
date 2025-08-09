# Just recipes for PCP Pipe

set shell := ["/bin/sh", "-cu"]

default:
    @just --list

# Install cargo-nextest if missing (optional convenience)
setup:
    cargo install cargo-nextest --locked --version ^0.9 || true

# Build and checks
build:
    cargo build --workspace --locked

check:
    cargo check --workspace

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace -- -D warnings

# Tests (use nextest)
test:
    cargo nextest run --workspace

test-ci:
    cargo nextest run --workspace --profile ci

# Run CLI
run url="https://overcast.fm/+AAA6x97Mzdc":
    cargo run -p pcp-cli -- {{url}}

gpu-metal-build:
    cargo build -F pcp-transcribe/gpu-metal

gpu-openmp-build:
    cargo build -F pcp-transcribe/gpu-openmp

clean:
    cargo clean

update:
    cargo update -w
