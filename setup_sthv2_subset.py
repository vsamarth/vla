import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

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
