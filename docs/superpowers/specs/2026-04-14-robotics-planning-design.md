# Design Doc: Robotics Planning with Gemma 4

A multimodal script that uses Gemma 4 to analyze a robotics scene from an image and generate a natural language plan to achieve a specific goal.

## 1. Goal
To demonstrate Gemma 4's multimodal capabilities by providing it with a visual scene from a robotics dataset (e.g., RT-2 or BridgeData) and a goal, then having it output a coherent step-by-step plan for a robot arm.

## 2. Architecture
- **Language:** Python 3.10+
- **Package Manager:** `uv`
- **Libraries:**
    - `transformers`: To load the Gemma 4 model and its multimodal processor.
    - `torch`: Deep learning backend (optimized for CUDA).
    - `Pillow` (PIL): To load and process images.
    - `requests`: To fetch sample images from public robotics datasets.
- **Model:** `google/gemma-4-e4b-it` (Effective 4B with native vision-language support).

## 3. Implementation Details
- **Project Structure:**
    - `robotics_planner.py`: The main execution script.
- **Logic Flow:**
    1. Check for CUDA and `HF_TOKEN`.
    2. Load the multimodal `Processor` and the `Model` from the Gemma 4 ID.
    3. Fetch a sample image from a known robotics benchmark (e.g., a "tabletop with blocks" scene).
    4. Define a goal (e.g., "Put the green block in the red bowl").
    5. Construct a prompt that includes the image and the task.
    6. Use the processor to format the input for the model.
    7. Generate a natural language response.
    8. Print the generated plan.

## 4. Usage
```bash
export HF_TOKEN="your_token"
uv run robotics_planner.py
```

## 5. Success Criteria
- The script successfully fetches and loads the scene image.
- Gemma 4 generates a relevant, step-by-step plan that references objects in the image.
- The model correctly handles the multimodal (Image + Text) input.
