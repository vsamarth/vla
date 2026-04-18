#!/usr/bin/env python3
"""
CALVIN to LAPA Data Converter

Converts CALVIN dataset to LAPA's JSONL format for fine-tuning.

LAPA expects JSONL with per-row format:
{
    "instruction": "<s> You are a helpful assistant. USER: {lang} ASSISTANT:",
    "vision": [256 vqgan tokens],  # pre-encoded as integers
    "action": [7 discretized action tokens],  # bin indices
    "raw_actions": [7 float actions],  # for loss computation
    "fields": "[instruction],[vision],action"
}
"""

import os
import sys
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

# Add LAPA to path
LAPA_ROOT = Path(__file__).parent.parent.parent / "LAPA"
sys.path.insert(0, str(LAPA_ROOT))

from latent_pretraining.vqgan import VQGAN
from latent_pretraining.data import VisionActionProcessor


class CalvinToLAPAConverter:
    def __init__(
        self,
        calvin_root: str,
        lapa_root: str,
        output_dir: str,
        discretization_bins: int = 256,
    ):
        """
        Args:
            calvin_root: Path to CALVIN repository
            lapa_root: Path to LAPA repository
            output_dir: Where to save converted data
            discretization_bins: Number of bins for action discretization
        """
        self.calvin_root = Path(calvin_root)
        self.lapa_root = Path(lapa_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.discretization_bins = discretization_bins

        # Preprocessor for images (resize to 256x256)
        self.preprocessor = A.Compose(
            [
                A.LongestMaxSize(max_size=256),
                A.Resize(256, 256),
            ]
        )

        # VQGAN for encoding images
        vqgan_path = self.lapa_root / "lapa_checkpoints" / "vqgan"
        if vqgan_path.exists():
            print(f"Loading VQGAN from {vqgan_path}")
            self.vqgan = VQGAN(str(vqgan_path), replicate=False)
            self.vqgan_loaded = True
        else:
            print(f"WARNING: VQGAN checkpoint not found at {vqgan_path}")
            print("Vision tokens will need to be pre-encoded another way")
            self.vqgan = None
            self.vqgan_loaded = False

        # Tokenizer for instructions
        tokenizer_path = self.lapa_root / "lapa_checkpoints" / "tokenizer.model"
        if tokenizer_path.exists():
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor(str(tokenizer_path))
            self.tokenizer_loaded = True
        else:
            print(f"WARNING: Tokenizer not found at {tokenizer_path}")
            self.tokenizer = None
            self.tokenizer_loaded = False

        # Action bins (fitted during conversion)
        self.action_bins = None

    def download_debug_dataset(self):
        """Download the CALVIN debug dataset if not present."""
        dataset_dir = self.calvin_root / "dataset"

        if (dataset_dir / "calvin_debug_dataset" / "training").exists():
            print("Debug dataset already exists, skipping download.")
            return

        zip_path = dataset_dir / "calvin_debug_dataset.zip"

        if not zip_path.exists():
            print("Downloading CALVIN debug dataset...")
            subprocess.run(
                [
                    "wget",
                    "-q",
                    "--show-progress",
                    "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip",
                ],
                cwd=dataset_dir,
                check=True,
            )

        print("Unzipping...")
        subprocess.run(
            ["unzip", "-q", "calvin_debug_dataset.zip"], cwd=dataset_dir, check=True
        )

        # Clean up
        if zip_path.exists():
            zip_path.unlink()

        print("Download complete!")

    def load_episodes(self, split: str = "training") -> List[Dict]:
        """Load all episodes from a split."""
        dataset_dir = self.calvin_root / "dataset" / "calvin_debug_dataset"
        split_dir = dataset_dir / split

        episodes = []
        for npz_path in sorted(split_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            episode = {key: data[key] for key in data.files}
            episodes.append(episode)

        return episodes

    def load_language_annotations(self, split: str = "training") -> Dict:
        """Load language annotations for a split."""
        dataset_dir = self.calvin_root / "dataset" / "calvin_debug_dataset"
        lang_dir = split_dir = dataset_dir / split / "lang_annotations"

        ann_path = lang_dir / "auto_lang_ann.npy"
        if ann_path.exists():
            return np.load(ann_path, allow_pickle=True).item()
        return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to 256x256 and normalize."""
        # PIL expects RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        processed = self.preprocessor(image=image)["image"]
        return processed.astype(np.float32) / 127.5 - 1.0

    def encode_image(self, image: np.ndarray) -> List[int]:
        """Encode a single image to VQGAN tokens."""
        if not self.vqgan_loaded:
            return [0] * 256

        image_batch = np.expand_dims(image, 0)
        encoded = self.vqgan.encode(image_batch)
        codebook_indices = encoded[1]
        tokens = codebook_indices.flatten().tolist()
        return tokens

    def compute_action_bins(self, episodes: List[Dict]):
        """Fit action bins from all episodes."""
        print("Computing action bins...")

        all_actions = []
        for episode in episodes:
            if "actions" in episode:
                a = episode["actions"]
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                all_actions.append(a)
            if "rel_actions" in episode:
                a = episode["rel_actions"]
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                all_actions.append(a)

        if not all_actions:
            print("WARNING: No actions found in episodes")
            return

        all_actions = np.concatenate(all_actions, axis=0)
        print(f"Collected {all_actions.shape} action samples")

        if all_actions.ndim == 1:
            all_actions = all_actions.reshape(-1, 7)

        # Fit bins per dimension
        bins = {}
        for i in range(7):
            try:
                _, bin_edges = pd.qcut(
                    all_actions[:, i],
                    self.discretization_bins,
                    labels=False,
                    retbins=True,
                    duplicates="drop",
                )
                bins[i] = bin_edges.tolist()
            except Exception as e:
                print(f"Warning: Could not discretize dimension {i}: {e}")
                # Use uniform bins
                bins[i] = np.linspace(
                    all_actions[:, i].min(),
                    all_actions[:, i].max(),
                    self.discretization_bins + 1,
                ).tolist()

        self.action_bins = bins

        # Save bins as CSV (each dimension has variable length bins)
        with open(self.output_dir / "action_bins.csv", "w") as f:
            for i in range(7):
                f.write(f"{i}," + ",".join(map(str, bins[i])) + "\n")
        print(f"Action bins saved to {self.output_dir / 'action_bins.csv'}")

        return bins

    def normalize_action(self, raw_action) -> np.ndarray:
        """Normalize raw action to 7-element numpy array."""
        if raw_action is None:
            return np.zeros(7, dtype=np.float32)

        if isinstance(raw_action, (int, float)):
            arr = np.array([float(raw_action)], dtype=np.float32)
        elif isinstance(raw_action, np.ndarray):
            arr = raw_action.astype(np.float32).flatten()
        else:
            arr = np.array(raw_action, dtype=np.float32).flatten()

        if len(arr) < 7:
            arr = np.pad(arr, (0, 7 - len(arr)))
        elif len(arr) > 7:
            arr = arr[:7]

        return arr

    def discretize_action(self, action: np.ndarray) -> List[int]:
        """Convert continuous action to discretized bin indices."""
        if self.action_bins is None:
            raise ValueError("Action bins not computed yet")

        action = self.normalize_action(action)

        result = []
        for i in range(7):
            bins = self.action_bins[i]
            val = float(action[i])
            for j in range(len(bins) - 1):
                if bins[j] <= val < bins[j + 1]:
                    result.append(j)
                    break
            else:
                result.append(len(bins) - 2)

        return result

    def convert_episode(
        self,
        episode: Dict,
        lang_ann: Optional[Dict] = None,
        episodes_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """Convert a single episode to LAPA format."""

        n_timesteps = len(episode.get("actions", episode.get("rel_actions", [])))

        if n_timesteps == 0:
            return []

        # Get language annotations if available
        lang_texts = None
        if lang_ann is not None:
            lang_texts = lang_ann.get("language", {}).get("ann", [None] * n_timesteps)

        converted_rows = []

        for t in range(n_timesteps):
            row = {}

            # Language instruction
            if lang_texts is not None and t < len(lang_texts) and lang_texts[t]:
                instruction = (
                    f"<s> You are a helpful assistant. USER: {lang_texts[t]} ASSISTANT:"
                )
            else:
                instruction = "<s> You are a helpful assistant. USER: perform the next action ASSISTANT:"

            row["instruction"] = instruction

            # Vision tokens (pre-encode image)
            rgb_static = episode.get("rgb_static", None)
            if rgb_static is not None and t < len(rgb_static):
                img = rgb_static[t]
                processed = self.preprocess_image(img)
                vision_tokens = self.encode_image(processed)
            else:
                vision_tokens = [0] * 256

            row["vision"] = vision_tokens

            # Actions
            action_key = "rel_actions" if "rel_actions" in episode else "actions"
            if action_key in episode and t < len(episode[action_key]):
                raw_action = self.normalize_action(episode[action_key][t])
            else:
                raw_action = np.zeros(7, dtype=np.float32)

            row["raw_actions"] = raw_action.tolist()

            # Discretized action (if bins are computed)
            if self.action_bins is not None:
                row["action"] = self.discretize_action(np.array(raw_action))
            else:
                # Placeholder if bins not yet computed
                row["action"] = [0] * 7

            row["fields"] = "[instruction],[vision],action"

            converted_rows.append(row)

        return converted_rows

    def convert_split(self, split: str = "training"):
        """Convert a full split (training or validation)."""
        print(f"Converting {split} split...")

        # Load episodes
        episodes = self.load_episodes(split)
        print(f"Loaded {len(episodes)} episodes")

        # Compute action bins from training split
        if self.action_bins is None and split == "training":
            self.compute_action_bins(episodes)

        # Load language annotations
        lang_ann = self.load_language_annotations(split)

        # Convert all episodes
        all_rows = []
        for i, episode in enumerate(episodes):
            rows = self.convert_episode(episode, lang_ann)
            all_rows.extend(rows)
            if (i + 1) % 10 == 0:
                print(f"  Converted {i + 1}/{len(episodes)} episodes...")

        print(f"Converted {len(all_rows)} total timesteps")

        # Save as JSONL
        output_path = self.output_dir / f"calvin_{split}.jsonl"
        with open(output_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        print(f"Saved to {output_path}")

        return all_rows

    def verify_output(self):
        """Verify the converted data."""
        print("\n=== Verification ===")

        for split in ["training", "validation"]:
            path = self.output_dir / f"calvin_{split}.jsonl"
            if path.exists():
                with open(path) as f:
                    lines = f.readlines()

                print(f"\n{split}:")
                print(f"  Samples: {len(lines)}")

                if lines:
                    sample = json.loads(lines[0])
                    print(f"  Keys: {list(sample.keys())}")
                    print(f"  Vision tokens: {len(sample.get('vision', []))}")
                    print(f"  Action: {sample.get('action', [])}")
                    print(f"  Raw actions: {sample.get('raw_actions', [])}")
            else:
                print(f"\n{split}: NOT FOUND")

        bins_path = self.output_dir / "action_bins.csv"
        if bins_path.exists():
            print(f"\nAction bins: {bins_path}")
            with open(bins_path) as f:
                lines = f.readlines()
            print(f"  Dimensions: {len(lines)}")
            print(f"  Sample bins (dim 0): {len(lines[0].split(',')) - 1} bins")

        print("=== End Verification ===\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CALVIN dataset to LAPA format"
    )
    parser.add_argument(
        "--calvin_root",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "calvin"),
    )
    parser.add_argument(
        "--lapa_root",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "LAPA"),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--discretization_bins", type=int, default=256)
    parser.add_argument(
        "--verify", action="store_true", help="Only verify existing output"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "data"

    converter = CalvinToLAPAConverter(
        calvin_root=args.calvin_root,
        lapa_root=args.lapa_root,
        output_dir=args.output_dir,
        discretization_bins=args.discretization_bins,
    )

    if args.verify:
        converter.verify_output()
    else:
        # Download if needed
        converter.download_debug_dataset()

        # Convert both splits
        converter.convert_split("training")
        converter.convert_split("validation")

        # Verify
        converter.verify_output()


if __name__ == "__main__":
    main()
