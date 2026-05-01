# CALVIN to LAPA Fine-tuning

This repository contains scripts for processing the CALVIN dataset and fine-tuning the LAPA (Latent Action Pretraining) model.

## Scripts Overview

### 1. `setup_calvin_lapa.sh`
Sets up the entire environment for fine-tuning.
- Clones necessary repositories (CALVIN, LAPA).
- Downloads the CALVIN ABC dataset (~517 GB).
- Downloads LAPA checkpoints (VQGAN, Tokenizer).
- Initializes a Python virtual environment using `uv` and installs dependencies.
- Runs an initial data conversion check.

### 2. `calvin_to_lapa.py`
The primary data processing and feature generation script.
- **Vision Features**: Encodes CALVIN images into 256 discrete tokens using a pre-trained VQGAN.
- **Action Features**: Discretizes 7-DoF continuous actions into bins (default: 256 bins) using a `qcut` strategy.
- **Format**: Converts the CALVIN `.npz` episodes into LAPA-compatible `.jsonl` files.

**Usage:**
```bash
python calvin_to_lapa.py --dataset task_ABC_D --output_dir ./data
```

### 3. `finetune_calvin.sh`
Launches the fine-tuning process on the processed dataset.
- Configures the JAX/Flax environment.
- Points to the generated `.jsonl` files and action bins.
- Handles model loading and training loops.

### 4. `extract_hidden_states.py` (in `LAPA/scripts/`)
Used for advanced feature extraction from the frozen LAPA transformer.
- **Hidden States**: Extracts vectors for single positions, autoregressive action tokens, and all 32 transformer layers.
- **Vision Pooling**: Generates mean-pooled features from vision tokens.
- **Output**: Produces `.npy` files and metadata for probing or analysis.

**Usage:**
```bash
python ../../LAPA/scripts/extract_hidden_states.py \
    --input_jsonl data/calvin_train.jsonl \
    --checkpoint_path path/to/lapa_checkpoint \
    --output_dir ./features
```

## Data Pipeline
1. **Setup**: Run `bash setup_calvin_lapa.sh`.
2. **Process**: Run `python calvin_to_lapa.py` to generate the training features.
3. **Analyze (Optional)**: Use `extract_hidden_states.py` to extract transformer representations.
4. **Fine-tune**: Run `bash finetune_calvin.sh`.

## Requirements
- Python 3.10+
- `uv` for dependency management.
- JAX with CUDA support.
