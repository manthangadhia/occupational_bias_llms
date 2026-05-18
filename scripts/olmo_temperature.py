import json
import os
import sys
from pathlib import Path
import time
from scipy.stats import entropy
import argparse

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils import load_prompts_for_model, load_model, generate, generate_with_entropy, cleanup_model

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

TEMPERATURES = [0.2, 0.5, 0.7, 1.0]
PROMPT_CASES = ["given", "assumed"]

def main(track_entropy: bool = True,
     multigen: bool = True,
     num_prompts: int = 50,
     temperature: float = None,
     prompt_case: str = None
    ):
    """Run analysis for selected models with optional entropy tracking."""

    if temperature is None:
        temperature = TEMPERATURES

    if prompt_case:
        selected_prompt_cases = [prompt_case]
        output_name = f"olmo7b_temp_results_{prompt_case}.jsonl"
    else:
        selected_prompt_cases = PROMPT_CASES
        output_name = "olmo7b_temp_results.jsonl"

    with open(output_dir / output_name, "w", encoding="utf-8") as out_file:
        for model_key in MODELS.keys(): 
            # Load each model only once, then generate for all prompts and temperatures
            model_name = MODELS[model_key]
            print(f"\nPreparing to load model: {model_name}")
            model_load_start = time.time()
            tokenizer, model = load_model(model_name, cache_dir=models_dir)
            model_load_end = time.time()
            print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
            model.eval()

            # Start timing for this model
            model_start_time = time.time()

            # Load and configure prompts once per model and prompt case
            model_type = 'base' if model_key == 'base' else 'instruct'
            for prompt_case in selected_prompt_cases:
                prompts_df = load_prompts_for_model(model_type, prompt_case, limit=num_prompts)
                print(f"Loaded {len(prompts_df)} prompts for model '{model_key}' and case '{prompt_case}'")

                prompt_columns = set(prompts_df.columns)
                id_column = "id" if "id" in prompt_columns else ("profile_id" if "profile_id" in prompt_columns else None)
                prompt_column = "prompt" if "prompt" in prompt_columns else ("prompt_text" if "prompt_text" in prompt_columns else None)
                if prompt_column is None:
                    raise KeyError(f"Prompt text column not found in {sorted(prompt_columns)}")
                
                # Middle loop to go through all temperatures
                for temp in temperature:
                    print(f"\n[{model_key}] Generating at temperature: {temp}")
                    temp_start_time = time.time()

                    # Inner loop to go through all prompts
                    for prompt_data in prompts_df.itertuples(index=False):
                        profile_id = getattr(prompt_data, id_column) if id_column else "unknown"
                        prompt_text = getattr(prompt_data, prompt_column)
                        print(f"[{model_key}] Processing prompt for profile {profile_id}")
                        # Start tracking output for this prompt at this stage
                        model_output = {
                            "model_key": model_key,
                            "model_name": model_name,
                            "prompt_case": prompt_case,
                            "profile_id": profile_id,
                            "temperature": temp
                        }

                        for optional_field in ("gender", "occupation", "attended_university"):
                            if optional_field in prompt_columns:
                                model_output[optional_field] = getattr(prompt_data, optional_field)

                        if not multigen:
                            num_gens = 1
                        elif prompt_case == "assumed":
                            num_gens = 10
                        else:
                            num_gens = NUM_GENERATIONS
                        
                        for n in range(1, num_gens + 1):

                            if track_entropy: # generate response with entropy tracking
                                result_entropy = generate_with_entropy(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    clip_input=True,
                                    temperature=temp
                                )

                                response = result_entropy['text']
                                mean_entropy = result_entropy['mean_entropy']
                                max_entropy = result_entropy['max_entropy']
                                min_entropy = result_entropy['min_entropy']
                                std_entropy = result_entropy['std_entropy']

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
                                response = generate(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    clip_input=True,
                                    temperature=temp
                                )
                                model_output.update({
                                    "response_number": n,
                                    "response": response,
                                })
                                    
                            # Write output for this prompt and generation
                            out_file.write(json.dumps(model_output) + "\n")

                        print(f"[{model_key}] Generated {num_gens} responses ✓")
                        # Flush the file buffer to ensure data is written to disk AFTER EACH PROMPT
                        out_file.flush()
                        os.fsync(out_file.fileno())

                    temp_end_time = time.time()
                    temp_elapsed = temp_end_time - temp_start_time
                    print(f"[{model_key}] Completed temperature {temp} in {temp_elapsed:.2f} seconds ({temp_elapsed/60:.2f} minutes)")

            # End timing for this model
            model_end_time = time.time()
            elapsed_time = model_end_time - model_start_time
            print(f"\n[{model_key}] Total generation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"[{model_key}] Results saved to disk ✓")
            # Cleanup model from memory
            model, tokenizer = cleanup_model(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Olmo-7B model analysis.")
    parser.add_argument("--num_prompts", 
                        type=int, 
                        default=None, 
                        help="Number of prompts to process per model.")
    parser.add_argument("--temperature",
                        type=float, 
                        default=None, 
                        help="Temperature for text generation.")
    parser.add_argument("--prompt_case",
                        type=str,
                        choices=PROMPT_CASES,
                        default=None,
                        help="Run only one prompt case (given or assumed).")
    args = parser.parse_args()
    main(num_prompts=args.num_prompts, prompt_case=args.prompt_case)