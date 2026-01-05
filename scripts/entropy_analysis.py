"""
Entropy Analysis Script

This script analyzes the entropy of model generations to measure uncertainty
at each token position during text generation. It uses the IFT model with
the model singleton for efficient GPU memory management.

Higher entropy indicates more uncertainty (probability spread across many tokens)
Lower entropy indicates more certainty (probability concentrated on few tokens)
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

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
IFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Generation parameters
MAX_NEW_TOKENS = 50
NUM_GENERATIONS = 5  # Number of generations per prompt for consistency analysis


def generate_with_entropy_tracking(prompt, model, tokenizer, device, max_new_tokens=50):
    """
    Generate text while tracking the entropy at each token position.
    
    Args:
        prompt: Input text prompt
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on (cuda or cpu)
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Dictionary containing:
        - text: the complete generated response
        - entropies: list of entropy values, one per generated token
        - tokens: list of the actual tokens generated
        - top_probs: for each position, the probabilities and tokens of top 5 candidates
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Store entropy values and tokens as we generate
    token_entropies = []
    generated_tokens = []
    top_probs_per_position = []
    
    # Generate token by token to inspect each distribution
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the model's output for the current sequence
            outputs = model(input_ids)
            
            # Get logits for the next token position
            next_token_logits = outputs.logits[:, -1, :]
            
            # Convert logits to probabilities using softmax
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Calculate entropy of this distribution
            prob_dist = probs.cpu().numpy()[0]
            token_entropy = entropy(prob_dist)
            token_entropies.append(float(token_entropy))
            
            # Store the top 5 most probable tokens for inspection
            top_5_probs, top_5_indices = torch.topk(probs, 5)
            top_probs_per_position.append({
                'probs': top_5_probs.cpu().numpy()[0].tolist(),
                'tokens': [tokenizer.decode([idx]) for idx in top_5_indices[0]]
            })
            
            # Sample the next token using multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(tokenizer.decode(next_token[0]))
            
            # Add the new token to sequence for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if we generate an end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the full generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return {
        'text': generated_text,
        'entropies': token_entropies,
        'tokens': generated_tokens,
        'top_probs': top_probs_per_position
    }


def analyze_single_prompt(prompt, model, tokenizer, device, max_new_tokens=50):
    """
    Analyze entropy patterns for a single prompt.
    
    Returns:
        Dictionary with analysis results
    """
    result = generate_with_entropy_tracking(
        prompt, model, tokenizer, device, max_new_tokens
    )
    
    analysis = {
        'prompt': prompt,
        'generated_text': result['text'],
        'avg_entropy': np.mean(result['entropies']),
        'max_entropy': np.max(result['entropies']),
        'min_entropy': np.min(result['entropies']),
        'std_entropy': np.std(result['entropies']),
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
            'entropy': ent,
            'top_candidates': top['tokens'][:3],
            'top_probs': top['probs'][:3]
        })
    
    return analysis


def analyze_multiple_generations(prompt, model, tokenizer, device, 
                                 num_generations=5, max_new_tokens=50):
    """
    Generate multiple responses to the same prompt to analyze consistency.
    
    Returns:
        Dictionary with cross-generation analysis
    """
    all_entropies = []
    generations = []
    
    for i in range(num_generations):
        result = generate_with_entropy_tracking(
            prompt, model, tokenizer, device, max_new_tokens
        )
        all_entropies.append(result['entropies'])
        generations.append({
            'text': result['text'],
            'mean_entropy': np.mean(result['entropies'])
        })
    
    # Position-wise entropy comparison
    min_length = min(len(e) for e in all_entropies)
    position_wise_stats = []
    
    for pos in range(min_length):
        entropies_at_pos = [gen_entropies[pos] for gen_entropies in all_entropies]
        position_wise_stats.append({
            'position': pos,
            'mean': np.mean(entropies_at_pos),
            'std': np.std(entropies_at_pos),
            'min': np.min(entropies_at_pos),
            'max': np.max(entropies_at_pos)
        })
    
    return {
        'prompt': prompt,
        'num_generations': num_generations,
        'generations': generations,
        'position_wise_stats': position_wise_stats,
        'overall_mean_entropy': np.mean([g['mean_entropy'] for g in generations]),
        'overall_std_entropy': np.std([g['mean_entropy'] for g in generations])
    }


def main():
    """
    Run entropy analysis sequence
    """
    # Initialize model singleton
    model_manager = ModelSingleton()
    model_manager.set_cache_dir(models_dir)
    device = model_manager.get_device()
    print(f"Using device: {device}")
    
    # Load BASE model
    print(f"\nLoading model: {BASE_MODEL}")
    tokenizer, model = model_manager.load_model(BASE_MODEL, model_key="base")
    model.eval()  # Set to evaluation mode
    
    # Example prompts with different expected entropy patterns
    test_prompts = [
        "The capital of France is",  # Very predictable completion
        "The meaning of life is",     # More open-ended, likely higher entropy
        "Once upon a time",           # Creative continuation, variable uncertainty
        "In the field of technology, women",  # Gender bias test
        "The doctor walked into the room and"  # Occupation scenario
    ]
    
    output_file = data_dir / "entropy_analysis_results.json"
    
    print("\n" + "=" * 80)
    print("ENTROPY ANALYSIS OF MODEL GENERATIONS")
    print("=" * 80)
    
    all_results = {
        'model_name': IFT_MODEL,
        'device': device,
        'max_new_tokens': MAX_NEW_TOKENS,
        'single_prompt_analyses': [],
        'multiple_generation_analyses': []
    }
    
    # Analyze each prompt once
    print("\n--- SINGLE GENERATION ANALYSIS ---")
    for prompt in test_prompts:
        print(f"\nAnalyzing prompt: '{prompt}'")
        analysis = analyze_single_prompt(
            prompt, model, tokenizer, device, MAX_NEW_TOKENS
        )
        
        print(f"  Generated: {analysis['generated_text']}")
        print(f"  Avg entropy: {analysis['avg_entropy']:.3f}")
        print(f"  Max entropy: {analysis['max_entropy']:.3f}")
        print(f"  Min entropy: {analysis['min_entropy']:.3f}")
        print(f"  Std entropy: {analysis['std_entropy']:.3f}")
        
        all_results['single_prompt_analyses'].append(analysis)
    
    # Analyze consistency across multiple generations
    print("\n\n" + "=" * 80)
    print("MULTIPLE GENERATION CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # Test with a subset of prompts for multiple generations
    multi_gen_prompts = [
        "The future of artificial intelligence",
        "In the field of engineering, women"
    ]
    
    for prompt in multi_gen_prompts:
        print(f"\nAnalyzing {NUM_GENERATIONS} generations for: '{prompt}'")
        analysis = analyze_multiple_generations(
            prompt, model, tokenizer, device, 
            num_generations=NUM_GENERATIONS,
            max_new_tokens=MAX_NEW_TOKENS
        )
        
        print(f"  Overall mean entropy: {analysis['overall_mean_entropy']:.3f}")
        print(f"  Overall std entropy: {analysis['overall_std_entropy']:.3f}")
        
        for i, gen in enumerate(analysis['generations'], 1):
            print(f"  Gen {i}: {gen['text'][:80]}... (entropy: {gen['mean_entropy']:.3f})")
        
        all_results['multiple_generation_analyses'].append(analysis)
    
    # Save results to JSON
    print(f"\n\nSaving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Clean up
    model_manager.unload_model()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
