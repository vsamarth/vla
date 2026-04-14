import os
import pytest
from setup_sthv2_subset import list_videos, download_video

def test_list_videos():
    videos = list_videos(limit=5)
    assert len(videos) == 5
    assert all(v.endswith('.webm') for v in videos)

def test_download_video(tmp_path):
    videos = list_videos(limit=1)
    video_path = download_video(videos[0], tmp_path)
    assert os.path.exists(video_path)
    assert video_path.endswith('.webm')

def test_extract_frames(tmp_path):
    from setup_sthv2_subset import list_videos, download_video, extract_frames
    videos = list_videos(limit=1)
    video_path = download_video(videos[0], tmp_path)
    
    output_dir = tmp_path / "frames"
    extract_frames(video_path, output_dir)
    
    frames = list(output_dir.glob("img_*.jpg"))
    assert len(frames) > 0
    assert (output_dir / "img_00000.jpg").exists()
