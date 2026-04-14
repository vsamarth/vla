import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def main():
    # Using SmolVLM-256M: Ultra-fast multimodal model under 500M params
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
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
    
    # Use float32 for CPU/MPS for better stability, bfloat16 for CUDA
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )

    # 2. Fetch Sample Image (e.g., from RT-2 Project Page)
    url = "https://robotics-transformer2.github.io/images/rt2_social.png" 
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
