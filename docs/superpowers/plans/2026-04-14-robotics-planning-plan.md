# Robotics Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script that uses Gemma 4 to generate a robotics plan from a scene image.

**Architecture:** A Python script using `transformers` to load Gemma 4 (multimodal), `Pillow` for image handling, and `requests` to fetch a sample image from a robotics benchmark.

**Tech Stack:** Python, `uv`, `torch`, `transformers`, `Pillow`, `requests`.

---

### Task 1: Add New Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `Pillow` and `requests`**

Run: `uv add Pillow requests`
Expected: `pyproject.toml` and `uv.lock` updated.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add dependencies for multimodal robotics planning"
```

### Task 2: Implement Multimodal Robotics Planner

**Files:**
- Create: `robotics_planner.py`

- [ ] **Step 1: Write the implementation**

```python
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    model_id = "google/gemma-4-e4b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Processor and Model
    # Note: Using AutoModelForVision2Seq for multimodal Gemma 4
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device
    )

    # 2. Fetch Sample Image (e.g., from RT-2 GitHub)
    url = "https://raw.githubusercontent.com/google-research/robotics_transformer/master/docs/rt2_teaser.gif" # Teaser image
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(f"Loaded image from: {url}")

    # 3. Prepare Prompt
    # Gemma 4 uses <image> token for multimodal input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the scene and provide a 3-step plan to pick up the yellow block and place it in the blue bowl."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # 4. Process Inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    # 5. Generate
    print("\nGenerating robotics plan...\n")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512
    )

    # 6. Decode and Print
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Gemma 4 Plan:\n{response}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to verify**

Run: `uv run robotics_planner.py`
Expected: Image is fetched, model loads, and a plan is generated in natural language.

- [ ] **Step 3: Commit**

```bash
git add robotics_planner.py
git commit -m "feat: implement multimodal robotics planner with gemma 4"
```
