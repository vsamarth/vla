import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import cv2
from pathlib import Path

REPO_ID = "Nojah/limited_something_something_v2"

def list_videos(limit=100):
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    video_files = [f for f in files if f.startswith("videos/") and f.endswith(".webm")]
    return video_files[:limit]

def download_video(video_repo_path, local_dir):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=video_repo_path,
        repo_type="dataset",
        local_dir=local_dir
    )

def extract_frames(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"img_{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
    cap.release()
