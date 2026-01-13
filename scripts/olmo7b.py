import json
import os
import sys
from pathlib import Path
import time
from scipy.stats import entropy

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils import ModelSingleton, load_prompts_for_model

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
output_dir = data_dir / "olmo7b_results"
output_dir.mkdir(exist_ok=True)

# This looks for the "export" from your bash script
# If it doesn't find it, it uses root_dir / "models" as a backup
models_dir_path = os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models"))
models_dir = Path(models_dir_path)

print(f"Directing ModelSingleton to: {models_dir}")

# Model configuration
BASE_MODEL = "allenai/Olmo-3-1025-7B"
SFT_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
DPO_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
RLVR_MODEL = "allenai/Olmo-3-7B-Instruct"

MODELS = {
    "base": BASE_MODEL,
    "sft": SFT_MODEL,
    "dpo": DPO_MODEL,
    "rlvr": RLVR_MODEL,
}

# default generation parameters
MAX_NEW_TOKENS = 200
NUM_GENERATIONS = 5  # Number of generations per prompt for consistency analysis
DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
}


def main(track_entropy: bool = True, 
         multigen: bool = True, 
         num_prompts: int = 50
        ):
    """Run analysis for selected models with optional entropy tracking."""

    # Create model_manager
    model_manager = ModelSingleton()
    model_manager.set_cache_dir(models_dir)
    device = model_manager.get_device()
    print(f"Using device: {device}")

    with open(output_dir / "olmo7b_results.jsonl", "w", encoding="utf-8") as out_file:
        for model_key in MODELS.keys():
            model_name = MODELS[model_key]
            print(f"\nPreparing to load model: {model_name}")
            _, model = model_manager.load_model(model_name, model_key=model_key)
            model.eval()

            # Load and configure prompts
            model_type = 'base' if model_key == 'base' else 'instruct'
            prompts_to_process = load_prompts_for_model(model_type, limit=num_prompts)

            print(f"Loaded {len(prompts_to_process)} prompts for model '{model_key}'")
            
            # Start timing for this model
            model_start_time = time.time()

            for prompt_data in prompts_to_process:
                profile_id = prompt_data.get("profile_id", "unknown")
                prompt_text = prompt_data["prompt_text"]

                # Start tracking output for this prompt at this stage
                model_output = {
                    "model_key": model_key,
                    "model_name": model_name,
                    "profile_id": profile_id,
                }

                num_gens = NUM_GENERATIONS if multigen else 1
                
                for n in range(1, num_gens + 1):
                    print(f"[{model_key}] Generating response {n}/{num_gens} for profile {profile_id}...")

                    if track_entropy: # generate response with entropy tracking
                        result_entropy = model_manager.generate_with_entropy(
                            prompt=prompt_text,
                            clip_input=True,
                        )

                        response = result_entropy['text']
                        mean_entropy = result_entropy['mean_entropy']
                        max_entropy = result_entropy['max_entropy']
                        min_entropy = result_entropy['min_entropy']
                        std_entropy = result_entropy['std_entropy']
                        tokens = result_entropy['tokens']

                        model_output.update({
                            "response_number": n,
                            "response": response,
                            "entropy_analysis": {
                                "mean_entropy": mean_entropy,
                                "max_entropy": max_entropy,
                                "min_entropy": min_entropy,
                                "std_entropy": std_entropy,
                                },
                            })
                    
                    else: # generate response without entropy tracking
                        response = model_manager.generate(
                            prompt=prompt_text,
                            clip_input=True
                        )
                        model_output.update({
                            "response_number": n,
                            "response": response,
                        })
                            
                    # Write output for this prompt and generation
                    out_file.write(json.dumps(model_output) + "\n")
            
            # End timing for this model
            model_end_time = time.time()
            elapsed_time = model_end_time - model_start_time
            print(f"\n[{model_key}] Total generation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main(num_prompts=5)