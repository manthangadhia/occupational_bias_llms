"""
Generate Balanced Text with Entropy Analysis

This script combines text generation with entropy tracking. It generates multiple
responses per prompt while simultaneously measuring the entropy at each token position
during generation. Results are saved to separate files for generated text and entropy analysis.
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import entropy

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.model_singleton import ModelSingleton

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
models_dir = root_dir / "models"
models_dir.mkdir(exist_ok=True)

# Input/Output files
PROMPT_FILE = data_dir / "gender_prompts" / "balanced_prompts_gender_given.json"
OUTPUT_TEXT_FILE = data_dir / "output_with_entropy.jsonl"
OUTPUT_ENTROPY_FILE = data_dir / "entropy_analysis_with_metadata.json"

# Generation parameters
NUM_RESPONSES_PER_PROMPT = 10
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
IFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODELS = {
    "base": BASE_MODEL,
    "ift": IFT_MODEL
}


def generate_with_entropy_tracking(prompt, model, tokenizer, device, max_new_tokens=50, temperature=0.7):
    """
    Generate text while tracking the entropy at each token position.
    
    Args:
        prompt: Input text prompt
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on (cuda or cpu)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
    
    Returns:
        Dictionary containing:
        - text: the complete generated response
        - entropies: list of entropy values, one per generated token
        - tokens: list of the actual tokens generated
        - top_probs: for each position, the probabilities and tokens of top 5 candidates
    """
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = inputs.to(device)
    
    # Store entropy values and tokens as we generate
    token_entropies = []
    generated_token_ids = []
    top_probs_per_position = []

    # Manually implement KV-cache to bring memory usage to "normal levels"
    past_key_values = None
    
    # Generate token by token to inspect each distribution
    model.eval()
    with torch.no_grad():

        # First forward pass
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Then generate tokens one by one (auto-regressive generation)
        for _ in range(max_new_tokens):
            # Apply temperature to logits (smooth or sharpen distribution)
            logits = next_token_logits / temperature
            
            # Convert logits to probabilities using softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate entropy of this distribution
            prob_dist = probs.cpu().numpy()[0] # This numpy approach is slow, I'm copying the whole vocab each time
            token_entropy = entropy(prob_dist)
            # token_entropy = (-(probs * probs.log()).sum(dim=-1)).item() # Trying to just do computation on GPU
            token_entropies.append(float(token_entropy))
            
            # Store the top 5 most probable tokens for inspection
            top_5_probs, top_5_indices = torch.topk(probs, 5)
            top_probs_per_position.append({
                'probs': top_5_probs.cpu().numpy()[0].tolist(),
                'tokens': [tokenizer.decode([idx]) for idx in top_5_indices[0]]
            })
            
            # Sample the next token using multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)
            generated_token_ids.append(next_token.item()) # Track token IDs manually
            # generated_tokens.append(tokenizer.decode(next_token[0]))
            
            # Stop if we generate an end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Finally, feed only next_token
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
    
    # Decode the full generated sequence
    generated_text = (prompt + tokenizer.decode(generated_token_ids, skip_special_tokens=True))

    # Sanity check before returning
    assert len(token_entropies) == len(generated_token_ids), f"Mismatch in lengths of entropies ({len(token_entropies)}) and tokens ({len(generated_token_ids)})"
    
    return {
        'text': generated_text,
        'entropies': token_entropies,
        'tokens': tokenizer.convert_ids_to_tokens(generated_token_ids),
        'top_probs': top_probs_per_position
    }


def analyze_generation(prompt, result, profile_id, response_number, model_key):
    """
    Create analysis dictionary from generation result.
    
    Returns:
        Dictionary with analysis results including metadata
    """
    analysis = {
        'profile_id': profile_id,
        'response_number': response_number,
        'model': model_key,
        'prompt': prompt,
        'generated_text': result['text'],
        'avg_entropy': float(np.mean(result['entropies'])),
        'max_entropy': float(np.max(result['entropies'])),
        'min_entropy': float(np.min(result['entropies'])),
        'std_entropy': float(np.std(result['entropies'])),
        'num_tokens': len(result['tokens']),
        'token_details': []
    }
    
    # Add token-by-token details
    for i, (token, ent, top) in enumerate(zip(
        result['tokens'], 
        result['entropies'], 
        result['top_probs']
    )):
        analysis['token_details'].append({
            'position': i,
            'token': token,
            'entropy': float(ent),
            'top_candidates': top['tokens'][:3],
            'top_probs': [float(p) for p in top['probs'][:3]]
        })
    
    return analysis


def main():
    """
    Run generation with entropy analysis
    """
    # Initialize model singleton
    model_manager = ModelSingleton()
    model_manager.set_cache_dir(models_dir)
    device = model_manager.get_device()
    print(f"Using device: {device}")
    
    # clear any loaded models
    model_manager.unload_model()

    # Load prompts from JSON file
    print(f"\nLoading prompts from {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    
    # Separate prompts for each model
    ift_prompts = [p for p in all_prompts if p['key'] == 'ift']
    base_prompts = [p for p in all_prompts if p['key'] == 'base']
    prompts = {
        'ift': ift_prompts,
        'base': base_prompts,
    }
    print(f"Loaded {len(ift_prompts)} IFT prompts and {len(base_prompts)} base prompts")
    
    # Storage for all results
    all_entropy_analyses = []
    
    print("\n" + "=" * 80)
    print("GENERATION WITH ENTROPY ANALYSIS")
    print("=" * 80)

    bp = 1
    
    # Open output file for generated text
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as text_out:
        for model_key, model_name in MODELS.items():
            print(f"\n{'='*50}")
            print(f"Processing model: {model_key} ({model_name})")
            print(f"{'='*50}")
            
            # Load model using singleton
            tokenizer, model = model_manager.load_model(model_name, model_key=model_key)
            model.eval()  # Set to evaluation mode
            
            for prompt_data in prompts[model_key]:
                profile_id = prompt_data.get("profile_id", "unknown")
                prompt_text = prompt_data["prompt_text"]
                
                print(f"\n[{model_key}] Processing profile {profile_id}...")
                
                # Generate multiple responses for each prompt
                for response_num in range(1, NUM_RESPONSES_PER_PROMPT + 1):
                    print(f"  - Generating response {response_num}/{NUM_RESPONSES_PER_PROMPT}", end="")
                    
                    # Generate with entropy tracking
                    result = generate_with_entropy_tracking(
                        prompt_text,
                        model,
                        tokenizer,
                        device,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE
                    )
                    
                    # Extract just the response (remove the prompt)
                    response_text = result['text'][len(prompt_text):].strip()
                    
                    print(f" (avg entropy: {np.mean(result['entropies']):.3f})")
                    
                    # Save generated text to JSONL
                    text_entry = {
                        "profile_id": profile_id,
                        "model": model_key,
                        "model_name": model_name,
                        "response_number": response_num,
                        "prompt": prompt_text,
                        "response": response_text
                    }
                    text_out.write(json.dumps(text_entry) + "\n")
                    
                    # Create entropy analysis
                    entropy_analysis = analyze_generation(
                        prompt_text,
                        result,
                        profile_id,
                        response_num,
                        model_key
                    )
                    all_entropy_analyses.append(entropy_analysis)

                    if bp == 1:
                        break  # Break after first response for testing
                if bp == 1:
                    break  # Break after first prompt for testing
    
    # Save entropy analyses to JSON
    print(f"\n\nSaving entropy analysis to {OUTPUT_ENTROPY_FILE}")
    entropy_output = {
        'models': MODELS,
        'device': device,
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'num_responses_per_prompt': NUM_RESPONSES_PER_PROMPT,
        'analyses': all_entropy_analyses
    }
    
    with open(OUTPUT_ENTROPY_FILE, 'w', encoding='utf-8') as f:
        json.dump(entropy_output, f, indent=2, ensure_ascii=False)
    
    # Clean up
    model_manager.unload_model()
    
    print("\n" + "=" * 80)
    print("GENERATION AND ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Generated text saved to: {OUTPUT_TEXT_FILE}")
    print(f"Entropy analysis saved to: {OUTPUT_ENTROPY_FILE}")
    print(f"Total entries: {len(all_entropy_analyses)}")


if __name__ == "__main__":
    main()
