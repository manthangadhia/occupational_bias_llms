import json
import os
import sys
from pathlib import Path
import time
import argparse
import pandas as pd

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils import load_model, generate, generate_with_entropy, cleanup_model

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
output_dir = data_dir / "base_robustness_results"
output_dir.mkdir(exist_ok=True)
prompt_dir = data_dir / "robust_prompts"

# This looks for the "export" from your bash script
# If it doesn't find it, it uses root_dir / "models" as a backup
models_dir_path = os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models"))
models_dir = Path(models_dir_path)

print(f"Directing model cache to: {models_dir}")

# Base models under test.
# meta-llama/Llama-3.1-8B and google/gemma-2-9b are gated: HF_TOKEN (with
# access to both repos approved) must be set in .env for these to load.
MODELS = {
    "apertus-8b": "swiss-ai/Apertus-8B-2509",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma-2-9b": "google/gemma-2-9b",
}

# default generation parameters
MAX_NEW_TOKENS = 300
NUM_GENERATIONS = 10  # Number of generations per prompt for consistency analysis

TEMPERATURES = [0.2, 0.5, 0.7, 1.0]
PROMPT_STYLES = ["assumed", "author", "minimal", "neutral", "frog", "generic"]


def load_robustness_prompts(style: str, limit: int = 0) -> pd.DataFrame:
    """Load base-model robustness prompts for the given phrasing style ('assumed', 'author' or 'minimal')."""
    filepath = prompt_dir / f"{style}_prompt_base.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Robustness prompt file not found: {filepath}")

    prompts_df = pd.read_json(filepath)

    if limit and limit > 0:
        prompts_df = prompts_df[:limit]

    return prompts_df


def main(track_entropy: bool = True,
         multigen: bool = True,
         num_prompts: int = None,
         num_generations: int = NUM_GENERATIONS,
         temperature: float = None,
         prompt_style: str = None,
         model_key: str = None
        ):
    """Run robustness analysis for selected base models with optional entropy tracking."""

    if temperature is None:
        selected_temperature = TEMPERATURES
    else:
        selected_temperature = [temperature]

    if prompt_style:
        selected_styles = [prompt_style]
    else:
        selected_styles = PROMPT_STYLES

    if model_key:
        selected_models = {model_key: MODELS[model_key]}
    else:
        selected_models = MODELS

    output_name_parts = ["base_robustness_results"]
    if model_key:
        output_name_parts.append(model_key)
    if prompt_style:
        output_name_parts.append(prompt_style)
    if temperature:
        t_value = str(int(temperature * 10))
        output_name_parts.append(f"t{t_value}")
    
    output_name = "_".join(output_name_parts) + ".jsonl"

    with open(output_dir / output_name, "w", encoding="utf-8") as out_file:
        for model_key, model_name in selected_models.items():
            # Load each model only once, then generate for all prompts, styles and temperatures
            print(f"\nPreparing to load model: {model_name}")
            model_load_start = time.time()
            tokenizer, model = load_model(model_name, cache_dir=models_dir)
            model_load_end = time.time()
            print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
            model.eval()

            # Start timing for this model
            model_start_time = time.time()

            for style in selected_styles:
                prompts_df = load_robustness_prompts(style, limit=num_prompts)
                print(f"Loaded {len(prompts_df)} prompts for model '{model_key}' and style '{style}'")

                prompt_columns = set(prompts_df.columns)

                # Middle loop to go through all temperatures
                for temp in selected_temperature:
                    print(f"\n[{model_key}] Generating at temperature: {temp} (style: {style})")
                    temp_start_time = time.time()

                    # Inner loop to go through all prompts
                    for prompt_data in prompts_df.itertuples(index=False):
                        profile_id = prompt_data.id
                        prompt_text = prompt_data.prompt
                        print(f"[{model_key}] Processing prompt {profile_id} ({style})")

                        # Start tracking output for this prompt at this stage
                        model_output = {
                            "model_key": model_key,
                            "model_name": model_name,
                            "prompt_style": style,
                            "profile_id": profile_id,
                            "temperature": temp,
                        }

                        # Always emit the optional metadata fields so the JSONL schema stays
                        # consistent across styles. Styles without occupations (e.g. "frog",
                        # "generic") simply get None values.
                        for optional_field in ("occupation", "attended_university"):
                            model_output[optional_field] = (
                                getattr(prompt_data, optional_field)
                                if optional_field in prompt_columns
                                else None
                            )

                        num_gens = num_generations if multigen else 1

                        for n in range(1, num_gens + 1):

                            if track_entropy:  # generate response with entropy tracking
                                result_entropy = generate_with_entropy(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    clip_input=True,
                                    temperature=temp
                                )

                                response = result_entropy['text']
                                model_output.update({
                                    "response_number": n,
                                    "response": response,
                                    "entropy_analysis": {
                                        "mean_entropy": result_entropy['mean_entropy'],
                                        "mean_entropy_nucleus": result_entropy['mean_entropy_nucleus'],
                                    },
                                })

                            else:  # generate response without entropy tracking
                                response = generate(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    max_new_tokens=MAX_NEW_TOKENS,
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
            # Cleanup model from memory before loading the next one
            model, tokenizer = cleanup_model(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run base-model prompt-robustness analysis.")
    parser.add_argument("--num_prompts",
                        type=int,
                        default=None,
                        help="Number of prompts to process per model and style.")
    parser.add_argument("--num_generations",
                        type=int,
                        default=NUM_GENERATIONS,
                        help="Number of generations per prompt for consistency analysis.")
    parser.add_argument("--temperature",
                        type=float,
                        default=None,
                        help="Run only one temperature (default: sweep over all of TEMPERATURES).")
    parser.add_argument("--prompt_style",
                        type=str,
                        choices=PROMPT_STYLES,
                        default=None,
                        help="Run only one prompt class (assumed, author, or minimal).")
    parser.add_argument("--model",
                        type=str,
                        choices=sorted(MODELS.keys()),
                        default=None,
                        help="Run only one model (default: run all models sequentially).")
    args = parser.parse_args()
    main(num_prompts=args.num_prompts,
         num_generations=args.num_generations,
         temperature=args.temperature,
         prompt_style=args.prompt_style,
         model_key=args.model)
