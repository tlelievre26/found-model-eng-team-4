#!/usr/bin/env bash
set -euo pipefail

# Shared storage configuration
export FME_SHARED_DIR="/shared"
export HF_HOME="$FME_SHARED_DIR/models"

MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR="$HF_HOME/Qwen3-4B-Instruct-2507"

# Check shared directory permissions
if [[ ! -w "$HF_HOME" ]]; then
    echo "Shared model directory is not writable."
    echo "Please run setup_shared.sh first."
    exit 1
fi

# Create an isolated Python environment
python3 -m venv "$HOME/.venvs/hf-download"
source "$HOME/.venvs/hf-download/bin/activate"

# Install Hugging Face CLI
python -m pip install --upgrade huggingface_hub

# Download the base model
mkdir -p "$MODEL_DIR"

echo "Downloading $MODEL_ID..."

hf download "$MODEL_ID" \
    --local-dir "$MODEL_DIR"

# Make model files accessible to the team
chmod -R g+rwX "$MODEL_DIR"

# Verify downloaded files
echo "Model download complete."
echo "Model location: $MODEL_DIR"

du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR"
