import os
import random
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

from laq.laq_model import LatentActionQuantization


def load_video_frames(video_dir, frame_indices):
    frames = []
    for idx in frame_indices:
        frame_path = video_dir / f"img_{idx:05d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path).convert("RGB")
            frames.append(img)
        else:
            print(f"Warning: Frame {frame_path} not found, stopping")
            break
    return frames


def main():
    LAQ_CHECKPOINT = "laq_checkpoints/laq_openx.pt"
    FRAME_STEP = 5

    video_dirs = list(Path("data/sthv2_subset").iterdir())
    video_dir = random.choice(video_dirs)
    print(f"Selected video: {video_dir.name}")

    frame_indices = list(range(0, 40, FRAME_STEP))
    frames = load_video_frames(video_dir, frame_indices)

    if len(frames) < 2:
        print("Not enough frames in video")
        return

    print(f"Loaded {len(frames)} frames at indices: {frame_indices[: len(frames)]}")

    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )

    video_tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0)
    print(f"Video tensor shape: {video_tensor.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    laq = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=6,
        temporal_depth=6,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        device=device,
    )
    laq.load(LAQ_CHECKPOINT)
    laq.eval()

    output_dir = Path("output") / video_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    laq_codes = {}

    with torch.no_grad():
        for i in range(len(frames) - 1):
            first_idx = frame_indices[i]
            second_idx = frame_indices[i + 1]

            pair_tensor = video_tensor[:, :, i : i + 2].to(device)

            indices = laq.inference(pair_tensor, return_only_codebook_ids=True)
            laq_codes[f"{first_idx}_{second_idx}"] = indices.cpu().tolist()

            recon = laq.inference(pair_tensor, return_only_codebook_ids=False)
            recon_img = Image.fromarray(
                (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            recon_img.save(output_dir / f"decoded_frame_{second_idx:05d}.jpg")

            print(f"Pair ({first_idx}, {second_idx}): code={indices.cpu().tolist()}")

    with open(output_dir / "laq_codes.json", "w") as f:
        json.dump(laq_codes, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
