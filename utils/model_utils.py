"""
Utility functions for model loading, generation, and memory management.
This module provides a functional approach to model handling without singletons.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gc


def get_device() -> str:
    """Get the current device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def hard_cleanup_memory(*objects_to_delete, verbose: bool = True) -> Dict[str, Any]:
    """
    Perform aggressive memory cleanup and track memory usage.
    
    Args:
        *objects_to_delete: Variable number of objects to delete
        verbose: Whether to print memory statistics
        
    Returns:
        Dictionary with memory statistics before and after cleanup
    """
    device = get_device()
    stats = {}
    
    # Get memory before cleanup (if using CUDA)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        stats['memory_allocated_before_mb'] = torch.cuda.memory_allocated() / 1024**2
        stats['memory_reserved_before_mb'] = torch.cuda.memory_reserved() / 1024**2
    
    # Delete provided objects
    for obj in objects_to_delete:
        if obj is not None:
            del obj
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
        # Get memory after cleanup
        stats['memory_allocated_after_mb'] = torch.cuda.memory_allocated() / 1024**2
        stats['memory_reserved_after_mb'] = torch.cuda.memory_reserved() / 1024**2
        stats['memory_freed_mb'] = stats['memory_allocated_before_mb'] - stats['memory_allocated_after_mb']
        stats['reserved_freed_mb'] = stats['memory_reserved_before_mb'] - stats['memory_reserved_after_mb']
        
        if verbose:
            print(f"\n[Memory Cleanup Stats]")
            print(f"  Allocated: {stats['memory_allocated_before_mb']:.2f} MB → {stats['memory_allocated_after_mb']:.2f} MB")
            print(f"  Reserved:  {stats['memory_reserved_before_mb']:.2f} MB → {stats['memory_reserved_after_mb']:.2f} MB")
            print(f"  Freed:     {stats['memory_freed_mb']:.2f} MB (allocated), {stats['reserved_freed_mb']:.2f} MB (reserved)")
    else:
        if verbose:
            print("[Memory Cleanup] Running on CPU - CUDA stats not available")
    
    return stats


def load_model(model_name: str, cache_dir: Optional[Path] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load and return model + tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Optional cache directory for model downloads
        
    Returns:
        Tuple of (tokenizer, model)
    """
    device = get_device()
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    print(f"Model {model_name} loaded successfully on {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"Memory after load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    
    return tokenizer, model


def cleanup_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Tuple[None, None]:
    """
    Explicit cleanup of model and tokenizer with memory tracking.
    
    Args:
        model: The model to clean up
        tokenizer: The tokenizer to clean up
        verbose: Whether to print cleanup statistics
    """
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    return None, None


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    clip_input: bool = False
) -> str:
    """
    Generate text using the provided model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        clip_input: Whether to exclude input prompt from returned text
        
    Returns:
        Generated text response
    """
    device = get_device()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_tokens = output[0]
    if clip_input:
        # Decode only the newly generated tokens (excluding input prompt)
        generated_tokens = generated_tokens[input_length:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def generate_with_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    clip_input: bool = False,
    track_top_k: Optional[int] = None,
    return_per_token_entropy: bool = False
) -> dict:
    """
    Generate text while tracking entropy metrics.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        clip_input: Whether to exclude input prompt from returned text
        track_top_k: If set, store top-k token candidates per position (None = don't track)
        return_per_token_entropy: Whether to return the full list of per-token entropies
        
    Returns:
        dict with keys:
            - 'text': generated text (with or without prompt based on clip_input)
            - 'mean_entropy': average entropy across all generated tokens
            - 'max_entropy': maximum entropy value
            - 'min_entropy': minimum entropy value
            - 'std_entropy': standard deviation of entropy
            - 'num_tokens': number of tokens generated
            - 'per_token_entropy': list of entropy values (only if return_per_token_entropy=True)
            - 'tokens': list of generated token strings
            - 'top_k_data': per-position top-k info (only if track_top_k is set)
    """
    device = get_device()
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = inputs.to(device)
    
    # Storage for metrics
    token_entropies = []
    generated_token_ids = []
    top_k_data = [] if track_top_k else None
    
    # Manual generation with KV-cache
    past_key_values = None
    
    model.eval()
    with torch.no_grad():
        # First forward pass
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Apply temperature
            logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate entropy on GPU (more efficient than numpy)
            token_entropy = -torch.where(
                probs > 0,
                probs * probs.log(),
                torch.zeros_like(probs)
            ).sum(dim=-1).item()
            token_entropies.append(float(token_entropy))
            
            # Track top-k if requested
            if track_top_k:
                top_k_probs, top_k_indices = torch.topk(probs, track_top_k)
                # Move to CPU immediately
                top_k_probs_cpu = top_k_probs.cpu().numpy()[0].tolist()
                top_k_tokens_cpu = [tokenizer.decode([idx.item()]) for idx in top_k_indices[0].cpu()]
                # Delete GPU tensors immediately
                del top_k_probs, top_k_indices
                
                top_k_data.append({
                    'probs': top_k_probs_cpu,
                    'tokens': top_k_tokens_cpu
                })
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_token_ids.append(next_token.item())
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                del logits, probs, next_token
                break
            
            # Delete tensors before overwriting
            del logits, probs
            
            # Continue generation with KV-cache
            old_past_key_values = past_key_values
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Delete old KV cache and next_token
            del old_past_key_values, next_token
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    # Clip input prompt if requested
    if clip_input:
        final_text = generated_text
    else:
        final_text = prompt + generated_text
    
    # Build result dictionary
    result = {
        'text': final_text,
        'mean_entropy': float(np.mean(token_entropies)),
        'max_entropy': float(np.max(token_entropies)),
        'min_entropy': float(np.min(token_entropies)),
        'std_entropy': float(np.std(token_entropies)),
        'num_tokens': len(generated_token_ids),
        'tokens': [tokenizer.decode([token_id]) for token_id in generated_token_ids]
    }
    
    # Add optional fields
    if return_per_token_entropy:
        result['per_token_entropy'] = token_entropies
    
    if track_top_k:
        result['top_k_data'] = top_k_data

    # Hard cleanup for GPU memory
    hard_cleanup_memory(
        past_key_values,
        next_token_logits,
        outputs,
        input_ids,
        verbose=False  # Set to True to see memory stats after each generation
    )
    
    return result
