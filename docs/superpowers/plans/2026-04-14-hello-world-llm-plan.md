# Hello World LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a minimal, GPU-optimized "Hello World" LLM script using Hugging Face and `uv`.

**Architecture:** A single Python script using `transformers` to load a 0.5B parameter Qwen model, tokenize a prompt, and generate a response on CUDA.

**Tech Stack:** Python, `uv`, `torch`, `transformers`, `accelerate`.

---

### Task 1: Initialize Project and Dependencies

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Initialize the project with `uv`**

Run: `uv init`
Expected: `pyproject.toml` and `hello_llm.py` (boilerplate) created.

- [ ] **Step 2: Add dependencies**

Run: `uv add torch transformers accelerate`
Expected: `pyproject.toml` updated with dependencies.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: initialize project and add dependencies"
```

### Task 2: Implement GPU-Optimized LLM Script

**Files:**
- Modify: `hello_llm.py`

- [ ] **Step 1: Write the minimal implementation**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
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
```

- [ ] **Step 2: Run the script to verify**

Run: `uv run hello_llm.py`
Expected: Model downloads (if first run), loads onto GPU, and prints a coherent response.

- [ ] **Step 3: Commit**

```bash
git add hello_llm.py
git commit -m "feat: implement hello world llm script"
```
