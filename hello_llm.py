import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device
    )

    # 3. Prepare Prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you introduce yourself briefly?"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 4. Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 5. Generate
    print("\nGenerating response...\n")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 6. Decode
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
