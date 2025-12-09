"""
Model Singleton for efficient model loading and memory management.
Ensures only one model is loaded in memory at a time.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Tuple


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
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir: Optional[Path] = None
    
    def set_cache_dir(self, cache_dir: Path):
        """Set the cache directory for model downloads."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load a model and tokenizer. If a different model is already loaded,
        unload it first to free memory.
        
        Args:
            model_name: HuggingFace model identifier
            
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
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir=self.cache_dir
        )
        
        self.current_model_name = model_name
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
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            print("Model unloaded and memory cleared")
    
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
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
        
        # Decode only the newly generated tokens (*excluding* input prompt)
        generated_tokens = output[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def get_device(self) -> str:
        """Return the current device being used."""
        return self.device
