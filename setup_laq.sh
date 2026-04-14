#!/bin/bash
set -e

echo "Step 1: Installing uv..."
pip install uv

echo "Step 2: Syncing dependencies..."
uv sync

echo "Step 3: Creating checkpoints directory..."
mkdir -p laq_checkpoints

echo "Step 4: Fetching LAQ (VQ-VAE) model weights..."
# Using the resolve/main link for direct download
wget -O laq_checkpoints/laq_openx.pt https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/laq_openx.pt

echo "Setup complete!"
echo ""
echo "To run inference, you need a JSONL file with 'id', 'image', 'instruction', and 'vision' (VQGAN tokens) fields."
echo "Command to run inference:"
echo "uv run python laq/inference_sthv2.py \\"
echo "  --input_file path/to/your/input.jsonl \\"
echo "  --laq_checkpoint laq_checkpoints/laq_openx.pt \\"
echo "  --codebook_size 8 \\"
echo "  --code_seq_len 4 \\"
echo "  --window_size 10 \\"
echo "  --layer 6 \\"
echo "  --dist_number 1 \\"
echo "  --divider 1 \\"
echo "  --unshuffled_jsonl output_latent_actions.jsonl"
