# Design Spec: Automated sthv2 Data Setup

## Overview
This document outlines the design for an automated script `setup_sthv2_subset.py` that downloads a subset of the Something-Something V2 dataset from Hugging Face and organizes it into a frame-based structure compatible with the `ImageVideoDataset` class.

## Goals
- Automate the download of 100 Something-Something V2 videos.
- Extract frames from each video and save them as individual JPEGs.
- Organize the data into the structure: `data/sthv2_subset/<video_id>/img_XXXXX.jpg`.
- Ensure the process is fully runnable by a single script using `uv`.

## Approaches Considered
- **Hugging Face `datasets` Streaming:** Failed due to loading script restrictions in the standard 2026 `datasets` version.
- **Manual Mirror Download (Chosen):** Using `Nojah/limited_something_something_v2` which hosts individual `.webm` files. This is the most robust and fastest method for a small (100-video) subset.

## Design
### 1. Data Acquisition
- **Source:** Hugging Face repository `Nojah/limited_something_something_v2`.
- **Method:** Use `huggingface_hub`'s `hf_hub_download` to fetch `.webm` files from the `videos/` directory.
- **Count:** 100 videos.

### 2. Processing & Extraction
- **Tool:** OpenCV (`cv2`) for frame decoding.
- **Output Format:** JPEG (`img_XXXXX.jpg`) to match the naming convention in `laq/laq_model/data.py`.
- **Structure:**
  ```
  data/
  └── sthv2_subset/
      ├── 102148/
      │   ├── img_00000.jpg
      │   ├── img_00001.jpg
      │   └── ...
      ├── 103874/
      │   ├── img_00000.jpg
      │   └── ...
      └── ...
  ```

### 3. Implementation Details
- **Script:** `setup_sthv2_subset.py`
- **Dependencies:** `huggingface_hub`, `opencv-python`, `tqdm` (for progress).
- **Automation:** The script will create the `data/sthv2_subset` directory if it doesn't exist.

## Success Criteria
- 100 video folders created in `data/sthv2_subset`.
- Each folder contains sequential JPEG frames.
- `ImageVideoDataset(folder='data/sthv2_subset', ...)` can successfully load the data.

## Self-Review
- **Placeholder scan:** None.
- **Internal consistency:** Matches the expected frame-based structure.
- **Scope check:** Focused on the 100-video subset.
- **Ambiguity check:** Explicitly specifies the source repo and output structure.
