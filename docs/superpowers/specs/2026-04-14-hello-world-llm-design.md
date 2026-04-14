# Design Doc: Hello World LLM with Hugging Face

A simple, educational "Hello World" script using the Hugging Face `transformers` library and `uv` for package management, optimized for RunPod GPUs.

## 1. Goal
To provide a clear, step-by-step example of how to load a small, instruction-tuned LLM, tokenize a prompt, generate a response on a GPU, and decode the output.

## 2. Architecture
- **Language:** Python 3.10+
- **Package Manager:** `uv`
- **Libraries:**
    - `transformers`: Core HF library for models and tokenizers.
    - `torch`: Deep learning backend (with CUDA support).
    - `accelerate`: For efficient model loading and device placement.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Small, fast, and optimized for instruction-following).

## 3. Implementation Details
- **Project Structure:**
    - `pyproject.toml`: Managed by `uv` for dependencies.
    - `hello_llm.py`: The main execution script.
- **Logic Flow:**
    1. Check for CUDA availability.
    2. Load `AutoTokenizer` from the model ID.
    3. Load `AutoModelForCausalLM` and move it to `cuda`.
    4. Define a simple user prompt.
    5. Apply the model's chat template to format the prompt correctly for the "Instruct" version.
    6. Tokenize the formatted prompt.
    7. Generate output tokens using `model.generate()`.
    8. Decode and print the result.

## 4. Usage
The user will run the script using:
```bash
uv run hello_llm.py
```

## 5. Success Criteria
- The script successfully downloads and loads the model into GPU VRAM.
- The model generates a coherent response to a "Hello, how are you?" style prompt.
- GPU memory is utilized correctly during generation.
