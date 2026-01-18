"""
Model Singleton for efficient model loading and memory management.
Ensures only one model is loaded in memory at a time.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import gc


class ModelSingleton:
    """Singleton class to manage model loading and unloading."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.current_model_name: Optional[str] = None
        self.current_model_key: Optional[str] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir: Optional[Path] = None
    
    def set_cache_dir(self, cache_dir: Path):
        """Set the cache directory for model downloads."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_name: str, model_key: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load a model and tokenizer. If a different model is already loaded,
        unload it first to free memory.
        
        Args:
            model_name: HuggingFace model identifier
            model_key: Optional key to identify model type (e.g., 'base', 'ift')
            
        Returns:
            Tuple of (tokenizer, model)
        """
        # If the same model is already loaded, return it
        if self.current_model_name == model_name and self.model is not None:
            print(f"Model {model_name} already loaded, reusing...")
            return self.tokenizer, self.model
        
        # Unload previous model if exists
        if self.model is not None:
            print(f"Unloading previous model: {self.current_model_name}")
            self.unload_model()
        
        # Load new model
        print(f"Loading model through manager: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            cache_dir=self.cache_dir,
        )
        
        self.current_model_name = model_name
        self.current_model_key = model_key
        print(f"Model {model_name} loaded successfully on {self.device}")
        
        return self.tokenizer, self.model
    
    def unload_model(self):
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_name = None
            self.current_model_key = None

            # Force immediate garbage collection
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            
            print("Model unloaded and memory cleared")
    
    def generate(self, prompt: str, 
                 max_new_tokens: int = 300, 
                 temperature: float = 0.7, 
                 clip_input: bool = False) -> str:
        """
        Generate text using the currently loaded model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1] # Keep track of input prompt length to omit it from output
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_tokens = output[0]
        model_is_ift = self.current_model_key == "ift"
        # print(f"model is ift? {model_is_ift}")
        if model_is_ift or clip_input:
            # print(f"clipping input, model is ift: {model_is_ift}")
            # Decode only the newly generated tokens (*excluding* input prompt)
            generated_tokens = generated_tokens[input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def generate_with_entropy(
        self,
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
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = inputs.to(self.device)
        
        # Storage for metrics
        token_entropies = []
        generated_token_ids = []
        top_k_data = [] if track_top_k else None
        
        # Manual generation with KV-cache
        past_key_values = None
        
        self.model.eval()
        with torch.no_grad():
            # First forward pass
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Apply temperature
                logits = next_token_logits / temperature
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Calculate entropy on GPU (more efficient than numpy)
                # token_entropy = -(probs * probs.log()).sum(dim=-1).item() # this was always returnign nans because of 0-probs
                token_entropy = -torch.where(
                    probs > 0,
                    probs * probs.log(),
                    torch.zeros_like(probs)
                ).sum(dim=-1).item()
                token_entropies.append(float(token_entropy))
                
                # Track top-k if requested
                if track_top_k:
                    top_k_probs, top_k_indices = torch.topk(probs, track_top_k)
                    top_k_data.append({
                        'probs': top_k_probs.cpu().numpy()[0].tolist(),
                        'tokens': [self.tokenizer.decode([idx]) for idx in top_k_indices[0]]
                    })
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                generated_token_ids.append(next_token.item())
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Continue generation with KV-cache
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
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
            'tokens': [self.tokenizer.decode([token_id]) for token_id in generated_token_ids] #self.tokenizer.convert_ids_to_tokens(generated_token_ids)
        }
        
        # Add optional fields
        if return_per_token_entropy:
            result['per_token_entropy'] = token_entropies
        
        if track_top_k:
            result['top_k_data'] = top_k_data

        # Hard cleanup for GPU memory
        del past_key_values
        del logits
        del next_token_logits
        del outputs
        del input_ids 
        del probs
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
    
    def get_device(self) -> str:
        """Return the current device being used."""
        return self.device
