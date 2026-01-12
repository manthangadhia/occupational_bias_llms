import json
import sys
from pathlib import Path

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.model_singleton import ModelSingleton

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"

PROMPT_FILE = data_dir / "prompts.json" #TODO: provide proper path to file in "data/gender_prompts/" dir
OUTPUT_FILE = data_dir / "output_05b.jsonl"

# Number of prompts to process and responses per prompt
# NUM_PROMPTS = 10
NUM_RESPONSES_PER_PROMPT = 1

# BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# IFT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Qwen 2.5 models
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
IFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODELS = {
    "base": BASE_MODEL,
    "ift": IFT_MODEL
}


# -------------------------

# Initialize model singleton
model_manager = ModelSingleton()
print(f"Using device: {model_manager.get_device()}")


# -------------------------
# main loop
# -------------------------

def main(entropy_tracking: bool = False):
    print("="*80 + f"\nRunning with entropy tracking: {entropy_tracking}\n" + "="*80)
    # Load prompts from JSON file
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    # Separate prompts for each model
    ift_prompts = [p for p in all_prompts if p['key'] != 'base']
    base_prompts = [p for p in all_prompts if p['key'] == 'base']
    prompts = {
        'ift': ift_prompts,
        'base': base_prompts,
    }
    print(f"Loaded {len(ift_prompts) + len(base_prompts)} prompts from {PROMPT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for model_key, model_name in MODELS.items():
            print(f"\n{'='*50}")
            print(f"Processing model: {model_key} ({model_name})")
            print(f"{'='*50}")

            # Load model using singleton (automatically unloads previous model)
            model_manager.load_model(model_name, model_key=model_key)

            for prompt_data in prompts[model_key]:
                profile_id = prompt_data.get("profile_id", "unknown")
                prompt_text = prompt_data["prompt_text"]

                print(f"[{model_key}] Processing profile {profile_id}...")

                # Generate multiple responses for each prompt
                for response_num in range(1, NUM_RESPONSES_PER_PROMPT + 1):
                    print(f"  - Generating response {response_num}/{NUM_RESPONSES_PER_PROMPT}")

                    if not entropy_tracking:
                        response = model_manager.generate(
                            prompt=prompt_text,
                            max_new_tokens=300,
                            temperature=0.7,
                            clip_input=True
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

                    else:
                        result_entropy = model_manager.generate_with_entropy(
                            prompt=prompt_text,
                            max_new_tokens=300,
                            temperature=0.7,
                            clip_input=True,
                            track_top_k=None
                        )

                        response = result_entropy['text']
                        mean_entropy = result_entropy['mean_entropy']
                        max_entropy = result_entropy['max_entropy']
                        min_entropy = result_entropy['min_entropy']
                        std_entropy = result_entropy['std_entropy']
                        tokens = result_entropy['tokens']

                        entry = {
                            'profile_id': profile_id,
                            'response_number': response_num,
                            'model': model_key,
                            'entropy': {
                                'mean': mean_entropy,
                                'max': max_entropy,
                                'min': min_entropy,
                                'std': std_entropy,
                            },
                            'response': response,
                            }
                        
                        out.write(json.dumps(entry) + "\n\n")

        # Clean up after processing all models
        model_manager.unload_model()

    print(f"\nDone! Saved results to {OUTPUT_FILE}")
    print(f"Total entries: {len(MODELS) * len(all_prompts) * NUM_RESPONSES_PER_PROMPT}")


if __name__ == "__main__":
    main(entropy_tracking=True)