import json
import os
import shutil
import random

def create_sample():
    # 1. Create a dummy directory structure that the script expects
    # The script uses .split('/img_')[0] so we need that specific naming
    sample_dir = "sample_data/episode_0"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 2. Copy the existing img.png into the directory as img_00000.jpg and img_00010.jpg
    # The script looks for 'next_step' based on window_size (default 10)
    shutil.copy("img.png", os.path.join(sample_dir, "img_00000.jpg"))
    shutil.copy("img.png", os.path.join(sample_dir, "img_00010.jpg"))
    
    # 3. Create the JSONL entry
    # 'vision' must be 256 tokens (dummy values for now)
    dummy_vision = [random.randint(0, 1000) for _ in range(256)]
    
    entry = {
        "id": "episode_0_00000",
        "image": os.path.abspath(os.path.join(sample_dir, "img_00000.jpg")),
        "instruction": "Put the purple eggplant on the yellow towel.",
        "vision": dummy_vision
    }
    
    # 4. Write to sample_input.jsonl
    with open("sample_input.jsonl", "w") as f:
        f.write(json.dumps(entry) + "\n")
    
    print("Successfully created sample_input.jsonl")
    print(f"Sample images created in: {sample_dir}")

if __name__ == "__main__":
    create_sample()
