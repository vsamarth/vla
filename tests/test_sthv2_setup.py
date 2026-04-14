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
