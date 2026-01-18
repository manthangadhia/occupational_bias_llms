# utils/__init__.py
from .model_singleton import ModelSingleton
from .prompt_loader import load_prompts_for_model
from .model_utils import load_model, generate, generate_with_entropy, cleanup_model