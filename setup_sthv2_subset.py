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

def main(limit=100, base_output_dir="data/sthv2_subset"):
    videos = list_videos(limit=limit)
    print(f"Found {len(videos)} videos. Starting download and extraction...")
    
    base_output_dir = Path(base_output_dir)
    temp_download_dir = Path("temp_videos")
    temp_download_dir.mkdir(exist_ok=True)
    
    for video_repo_path in tqdm(videos):
        video_id = Path(video_repo_path).stem
        video_local_path = download_video(video_repo_path, temp_download_dir)
        
        video_frame_dir = base_output_dir / video_id
        extract_frames(video_local_path, video_frame_dir)
        
        # Clean up the video file to save space
        os.remove(video_local_path)

    if temp_download_dir.exists() and not any(temp_download_dir.iterdir()):
        temp_download_dir.rmdir()
    
    print(f"Setup complete! Data is in {base_output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/sthv2_subset")
    args = parser.parse_args()
    main(limit=args.limit, base_output_dir=args.output_dir)
