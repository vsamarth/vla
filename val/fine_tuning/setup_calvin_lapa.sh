#!/bin/bash
set -e

# ============================================
# CALVIN to LAPA Fine-tuning Setup Script
# ============================================
# This script sets up the environment for fine-tuning LAPA on CALVIN dataset
# Run: bash setup_calvin_lapa.sh
# ============================================

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Navigate up from val/fine_tuning to the repo root (val/fine_tuning -> val -> vla)
REPO_ROOT="$( cd -- "$( dirname -- "$SCRIPT_DIR/../.." )" &> /dev/null && pwd )"
CALVIN_ROOT="$REPO_ROOT/calvin"
LAPA_ROOT="$REPO_ROOT/LAPA"
FINE_TUNING_DIR="$SCRIPT_DIR"

echo "============================================"
echo "CALVIN to LAPA Fine-tuning Setup"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo "CALVIN root: $CALVIN_ROOT"
echo "LAPA root: $LAPA_ROOT"
echo "Fine-tuning dir: $FINE_TUNING_DIR"
echo "============================================"

# ============================================
# Step 1: Download CALVIN Debug Dataset
# ============================================
echo ""
echo "Step 1: Downloading CALVIN debug dataset..."

cd "$CALVIN_ROOT/dataset"

# Check if already downloaded
if [ -d "calvin_debug_dataset" ]; then
    echo "Debug dataset already exists, skipping download."
else
    echo "Downloading debug dataset (1.3 GB)..."
    wget -q --show-progress http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
    
    echo "Unzipping..."
    unzip -q calvin_debug_dataset.zip
    
    # Clean up zip
    rm calvin_debug_dataset.zip
    echo "Download and extraction complete."
fi

# Verify structure
if [ ! -d "calvin_debug_dataset/training" ] || [ ! -d "calvin_debug_dataset/validation" ]; then
    echo "ERROR: Expected training/validation folders not found"
    exit 1
fi

echo "Dataset structure:"
ls -la "$CALVIN_ROOT/dataset/calvin_debug_dataset/"

# ============================================
# Step 2: Install Python dependencies
# ============================================
echo ""
echo "Step 2: Installing Python dependencies..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    
    # Install core dependencies (--break-system-packages for externally-managed Python)
    uv pip install jax flax optax numpy Pillow albumentations ml-collections sentencepiece pandas --system --break-system-packages
    
    # Install LAPA requirements
    uv pip install -r "$LAPA_ROOT/requirements.txt" --system --break-system-packages || true
    
else
    echo "uv not found, using pip..."
    pip install --break-system-packages jax flax optax numpy Pillow albumentations ml-collections sentencepiece pandas
    
    # Install LAPA requirements if available
    if [ -f "$LAPA_ROOT/requirements.txt" ]; then
        pip install --break-system-packages -r "$LAPA_ROOT/requirements.txt" || true
    fi
fi

echo "Dependencies installed."

# ============================================
# Step 3: Check Python dependencies
# ============================================
echo ""
echo "Step 3: Verifying Python dependencies..."

# Check if required packages are available
python3 -c "
import sys
try:
    import jax
    import flax
    import numpy as np
    import PIL
    import albumentations
    print('Core dependencies (JAX, Flax, NumPy, Pillow, albumentations): OK')
except ImportError as e:
    print(f'Missing dependency: {e}')
    sys.exit(1)
"

# Check LAPA package
python3 -c "
import sys
sys.path.insert(0, '$LAPA_ROOT')
from latent_pretraining.vqgan import VQGAN
print('LAPA VQGAN module: OK')
"

echo "Dependencies check passed."

# ============================================
# Step 4: Check VQGAN checkpoint
# ============================================
echo ""
echo "Step 4: Checking VQGAN checkpoint..."

VQGAN_PATH="$LAPA_ROOT/lapa_checkpoints/vqgan"
if [ -d "$VQGAN_PATH" ]; then
    echo "VQGAN checkpoint found at $VQGAN_PATH"
else
    echo "WARNING: VQGAN checkpoint not found at $VQGAN_PATH"
    echo "Please download from https://huggingface.co/latent-action-pretraining/LAPA-7B-openx"
fi

# ============================================
# Step 5: Run data conversion
# ============================================
echo ""
echo "Step 5: Converting CALVIN data to LAPA format..."

cd "$FINE_TUNING_DIR"

python3 -c "
import sys
sys.path.insert(0, '$LAPA_ROOT')
sys.path.insert(0, '$CALVIN_ROOT/calvin_models')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU if available

from calvin_to_lapa import CalvinToLAPAConverter

converter = CalvinToLAPAConverter(
    calvin_root='$CALVIN_ROOT',
    lapa_root='$LAPA_ROOT',
    output_dir='$FINE_TUNING_DIR/data'
)

# Download debug dataset if not present
converter.download_debug_dataset()

# Convert training data
print('Converting training data...')
converter.convert_split('training')

# Convert validation data  
print('Converting validation data...')
converter.convert_split('validation')

# Generate action bins
print('Computing action bins...')
converter.compute_action_bins()

print('Conversion complete!')
print(f'Output: $FINE_TUNING_DIR/data/')
"

# ============================================
# Step 6: Verify output
# ============================================
echo ""
echo "Step 5: Verifying output..."

if [ -f "$FINE_TUNING_DIR/data/calvin_train.jsonl" ]; then
    TRAIN_LINES=$(wc -l < "$FINE_TUNING_DIR/data/calvin_train.jsonl")
    echo "Training data: $TRAIN_LINES samples"
else
    echo "WARNING: Training data not found"
fi

if [ -f "$FINE_TUNING_DIR/data/calvin_val.jsonl" ]; then
    VAL_LINES=$(wc -l < "$FINE_TUNING_DIR/data/calvin_val.jsonl")
    echo "Validation data: $VAL_LINES samples"
else
    echo "WARNING: Validation data not found"
fi

if [ -f "$FINE_TUNING_DIR/data/action_bins.csv" ]; then
    echo "Action bins: $FINE_TUNING_DIR/data/action_bins.csv"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Review the converted data in $FINE_TUNING_DIR/data/"
echo "2. Run: bash $FINE_TUNING_DIR/finetune_calvin.sh"
echo "============================================"
