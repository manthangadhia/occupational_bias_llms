import json
import sys
from pathlib import Path

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.model_singleton import ModelSingleton

# -------------------------
# configuration
# -------------------------
data_dir = root_dir / "data"
models_dir = root_dir / "models"
models_dir.mkdir(exist_ok=True)

PROMPT_FILE = data_dir / "gender_prompts" / "prompts_gender_given.json"
OUTPUT_FILE = data_dir / "output_ift.jsonl"

# Number of prompts to process and responses per prompt
NUM_PROMPTS = 5
NUM_RESPONSES_PER_PROMPT = 3

# BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# IFT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Qwen 2.5 models
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
IFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODELS = {
    # "base": BASE_MODEL,
    "ift": IFT_MODEL
}


# -------------------------

# Initialize model singleton
model_manager = ModelSingleton()
model_manager.set_cache_dir(models_dir)
print(f"Using device: {model_manager.get_device()}")


# -------------------------
# main loop
# -------------------------

def main():
    # Load prompts from JSON file
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    
    # Take first n prompts
    prompts_to_process = all_prompts[:NUM_PROMPTS]
    print(f"Loaded {len(prompts_to_process)} prompts from {PROMPT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for model_key, model_name in MODELS.items():
            print(f"\n{'='*50}")
            print(f"Processing model: {model_key} ({model_name})")
            print(f"{'='*50}")
            
            # Load model using singleton (automatically unloads previous model)
            model_manager.load_model(model_name)

            for prompt_data in prompts_to_process:
                profile_id = prompt_data.get("profile_id", "unknown")
                prompt_text = prompt_data["prompt_text"]
                
                print(f"[{model_key}] Processing profile {profile_id}...")

                # Generate multiple responses for each prompt
                for response_num in range(1, NUM_RESPONSES_PER_PROMPT + 1):
                    print(f"  - Generating response {response_num}/{NUM_RESPONSES_PER_PROMPT}")
                    
                    response = model_manager.generate(
                        prompt=prompt_text,
                        max_new_tokens=200,
                        temperature=0.7
                    )

                    entry = {
                        "profile_id": profile_id,
                        "model": model_key,
                        "model_name": model_name,
                        "response_number": response_num,
                        "prompt": prompt_text,
                        "response": response
                    }
                    out.write(json.dumps(entry) + "\n")

        # Clean up after processing all models
        model_manager.unload_model()

    print(f"\nDone! Saved results to {OUTPUT_FILE}")
    print(f"Total entries: {len(MODELS) * NUM_PROMPTS * NUM_RESPONSES_PER_PROMPT}")


if __name__ == "__main__":
    main()
