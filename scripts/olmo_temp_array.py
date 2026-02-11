import os
import sys
from pathlib import Path
import time
import argparse
import pandas as pd

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils import (
    load_prompts_for_model, 
    load_model, 
    generate, 
    generate_with_entropy, 
    cleanup_model
    )

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

PROMPT_CASES = [
    "given",
    "assumed"
]

# default generation parameters
MAX_NEW_TOKENS = 200
NUM_GENERATIONS_GIVEN = 5  # Number of generations per prompt for consistency analysis
NUM_GENERATIONS_ASSUMED = 10 # More generations for assumed_prompts to be able to get 10 gender guesses per prompt
DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
}

TEMPERATURES = [0.2, 0.5, 0.7, 1.0, 1.2]

def get_model_family(family: str) -> dict:
    """
    A standard function to access a different model family for testing when I need to. The default is olmo:7b but 
    later this can also be used to access olmo:32b, and Qwen2.5:0.5b for quick and dirty local testing for example.
    
    Args:
        family: This is a string param to choose which family of models to load
    
    Returns:
        MODELS: Dictionary containing the label and model_id (HF) for all models in the requested family
    """
    if family == 'qwen':
        # Model configuration
        BASE_MODEL = "Qwen/Qwen2.5-1.5B"
        IFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

        MODELS = {
            "base": BASE_MODEL,
            "ift": IFT_MODEL
        }
        
        return MODELS
    elif family == 'olmo':
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
        return MODELS
    else:
        raise ValueError("Choose between Qwen and Olmo model families!")

def main(track_entropy: bool = True, 
         multigen: bool = True, 
         temperature: float = None,
         model_key: str = None,
         MODELS: dict = None
        ):
    """Run analysis for selected models with optional entropy tracking."""

    # Make sure we have all the necessary inputs
    try:
        model_key is not None
    except:
        raise ValueError("Model key cannot be none!")
    try: 
        MODELS is not None
    except:
        raise ValueError("Model family dict cannot be none!")

    if temperature is None:
        temperature = TEMPERATURES
    else:
        # Insert temp value in list to not get an error later
        temperature = [temperature]

    # Collect outputs in memory; save to JSON once per prompt case
    output_rows_by_case = {case: [] for case in PROMPT_CASES}

    # Load specific model and run inference on it
    model_name = MODELS[model_key]
    print(f"\nPreparing to load model: {model_name}")
    model_load_start = time.time()
    tokenizer, model = load_model(model_name, cache_dir=models_dir)
    model_load_end = time.time()
    print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
    model.eval()

    # Load and configure each category of prompts once per model
    model_type = 'base' if model_key == 'base' else 'instruct'
    # Start timing generations for this model
    model_start_time = time.time()

    # After loading the model, loop through both prompt classes and generate for each
    for prompt_case in PROMPT_CASES:      
        given_prompts = True if 'given' in prompt_case else False
        prompts_df = load_prompts_for_model(model_type, prompt_case)

        print(f"Loaded {len(prompts_df)} prompts for model '{model_key}' and case '{prompt_case}'")
        
        # Start timing for this prompt_case
        case_start_time = time.time()
        
        # Middle loop to go through all temperatures
        for temp in temperature:
            print(f"\n[{model_key}] Generating at temperature: {temp}")
            temp_start_time = time.time()

            # Inner loop to go through all prompts
            for prompt_data in prompts_df.itertuples():
                profile_id = prompt_data.profile_id
                prompt_text = prompt_data.prompt_text
                print(f"[{model_key}] Processing prompt for profile {profile_id}")
                # Start tracking output for this prompt at this stage
                model_output = {
                    "model_key": model_key,
                    "model_name": model_name,
                    "profile_id": profile_id,
                    "temperature": temp,
                    "occupation_category": prompt_data.occupation_category,
                    "attended_university": prompt_data.attended_university,
                }
                # Include gender in the out file for the gender_given case
                if given_prompts:
                    model_output.update({
                        "gender": prompt_data.gender
                    })
                
                # Set the correct number of generations for the prompt_case (given=5, assumed=10)
                if not multigen:
                    num_gens = 1
                elif given_prompts:
                    num_gens = NUM_GENERATIONS_GIVEN
                else: # assumed_prompts
                    num_gens = NUM_GENERATIONS_ASSUMED
                
                # Now finally generate the requested number of responses for the current prompt (profile_id)
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
                            "mean_entropy": mean_entropy,
                            "max_entropy": max_entropy,
                            "min_entropy": min_entropy,
                            "std_entropy": std_entropy,
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
                            
                    # Track output for this prompt and generation
                    output_rows_by_case[prompt_case].append(model_output.copy())

                print(f"[{model_key}] Generated {num_gens} responses for gender {prompt_case} case ✓")

            temp_end_time = time.time()
            temp_elapsed = temp_end_time - temp_start_time
            print(f"[{model_key}] Completed temperature {temp} in {temp_elapsed:.2f} seconds ({temp_elapsed/60:.2f} minutes)")
        # End timing for how long it took to generate responses for this prompt_case
        case_end_time = time.time()
        case_time_elapsed = case_end_time - case_start_time
        print(f"[{prompt_case}] Completed prompt_case in {case_time_elapsed:.2f} seconds ({case_time_elapsed/60:.2f} minutes)")

        # Save results once per prompt case
        output_df = pd.DataFrame(output_rows_by_case[prompt_case])
        output_path = output_dir / f"olmo7b_{model_key}_{prompt_case}_all_temps.json"
        output_df.to_json(output_path, orient="records", force_ascii=True)
        print(f"[{model_key}] Results saved to {output_path} ✓")

    # End timing for this model
    model_end_time = time.time()
    elapsed_time = model_end_time - model_start_time
    print(f"\n[{model_key}] Total generation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    # Cleanup model from memory
    model, tokenizer = cleanup_model(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Olmo-7B model analysis.")
    parser.add_argument("--model_key",
                        type=str,
                        required=True,
                        help="Key for model to load")
    parser.add_argument("--model_family",
                        type=str,
                        default="qwen",
                        help="Name of the family of models to load")
    args = parser.parse_args()
    model_family = get_model_family(args.model_family)
    main(model_key=args.model_key, MODELS=model_family)