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
FINE_TUNING_DIR="$SCRIPT_DIR"

echo "============================================"
echo "CALVIN to LAPA Fine-tuning Setup"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo "Fine-tuning dir: $FINE_TUNING_DIR"
echo "============================================"

# ============================================
# Step 1: Ensure CALVIN and LAPA repos exist
# ============================================
echo ""
echo "Step 1: Checking repositories..."

cd "$REPO_ROOT"

# Clone CALVIN if not exists
if [ ! -d "calvin" ]; then
    echo "Cloning CALVIN repository..."
    git clone https://github.com/mees/calvin.git
    # Init submodules
    cd calvin && git submodule update --init --recursive && cd ..
else
    echo "CALVIN repository already exists"
fi

# Clone LAPA if not exists
if [ ! -d "LAPA" ]; then
    echo "Cloning LAPA repository..."
    git clone https://github.com/LatentActionPretraining/LAPA.git
else
    echo "LAPA repository already exists"
fi

CALVIN_ROOT="$REPO_ROOT/calvin"
LAPA_ROOT="$REPO_ROOT/LAPA"

echo "CALVIN root: $CALVIN_ROOT"
echo "LAPA root: $LAPA_ROOT"

# ============================================
# Step 2: Download CALVIN Debug Dataset
# ============================================
echo ""
echo "Step 2: Downloading CALVIN debug dataset..."

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
# Step 3: Download LAPA Checkpoints
# ============================================
echo ""
echo "Step 3: Checking LAPA checkpoints..."

mkdir -p "$LAPA_ROOT/lapa_checkpoints"

# Download from HuggingFace if not present
if [ ! -d "$LAPA_ROOT/lapa_checkpoints/vqgan" ]; then
    echo "Installing huggingface_hub..."
    pip install --break-system-packages huggingface_hub
    
    echo "Downloading LAPA checkpoints from HuggingFace..."
    python3 -c "
import os
from huggingface_hub import snapshot_download

checkpoint_path = '$LAPA_ROOT/lapa_checkpoints'
print('Downloading LAPA-7B-openx checkpoints...')
snapshot_download(
    repo_id='latent-action-pretraining/LAPA-7B-openx',
    local_dir=checkpoint_path,
    local_dir_use_symlinks=False
)
print('Download complete!')
"
else
    echo "LAPA checkpoints already exist"
fi

ls -la "$LAPA_ROOT/lapa_checkpoints/"

# ============================================
# Step 4: Install Python dependencies
# ============================================
echo ""
echo "Step 4: Installing Python dependencies..."

# Install all dependencies explicitly to avoid build failures
# Pin numpy<2 and scipy<1.17 for JAX compatibility
pip install --break-system-packages \
    'numpy<2' \
    Pillow \
    albumentations \
    pandas \
    'scipy<1.17' \
    transformers \
    datasets \
    tokenizers \
    sentencepiece \
    tux \
    flax \
    optax \
    chex \
    einops \
    tqdm \
    ml-collections \
    wandb \
    gcsfs \
    requests \
    jax \
    jaxlib \
    --no-build-isolation

echo "Dependencies installed."

# Verify transformers is installed
python3 -c "import transformers; print(f'transformers {transformers.__version__} installed')"

# ============================================
# Step 5: Verify Python dependencies
# ============================================
echo ""
echo "Step 5: Verifying Python dependencies..."

python3 -c "
import sys
missing = []
for mod in ['jax', 'flax', 'numpy', 'PIL', 'albumentations', 'sentencepiece', 'pandas', 'transformers']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f'Missing: {missing}')
    sys.exit(1)
print('Core dependencies: OK')
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
# Step 6: Run data conversion
# ============================================
echo ""
echo "Step 6: Converting CALVIN data to LAPA format..."

cd "$FINE_TUNING_DIR"

# Create data output directory
mkdir -p "$FINE_TUNING_DIR/data"

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
# Step 7: Verify output
# ============================================
echo ""
echo "Step 7: Verifying output..."

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
