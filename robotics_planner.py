import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import platform

def main():
    # Detect platform and choose model
    current_platform = platform.system().lower()
    
    if current_platform == "darwin": # Mac
        model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        print(f"Mac detected. Using small model: {model_id}")
    else: # Linux / Server
        model_id = "google/gemma-4-e2b-it"
        print(f"Linux/Server detected. Using powerful model: {model_id}")
    
    # Detect device: CUDA for NVIDIA, MPS for Mac GPU, else CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")

    # 1. Load Processor and Model
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Use bfloat16 for CUDA/MPS, float32 for CPU
    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device
    )

    # 2. Load Local Image (from img.png)
    image = Image.open("img.png").convert("RGB")
    print("Loaded local image: img.png")

    # 3. Prepare Prompt for Gemma 4
    # Gemma 4 supports thinking via natural language instruction in the prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Task: Put the purple eggplant on the yellow towel.\nThink step-by-step about the scene and what you see. Then, provide a concise 3-step robot execution plan."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    print("-" * 20)
    print("FULL FORMATTED PROMPT:")
    print(text)
    print("-" * 20)

    # 4. Process Inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    # 5. Generate
    print("\nGenerating robotics plan...\n")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1
    )

    # 6. Decode and Print ALL Output
    full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("FULL MODEL OUTPUT:")
    print("-" * 20)
    print(full_output)
    print("-" * 20)

if __name__ == "__main__":
    main()
