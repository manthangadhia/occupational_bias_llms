# utils/prompt_loader.py
import json
from pathlib import Path
from typing import List, Dict

# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"

def load_prompts_for_model(model_type: str, 
                           prompt_file: str = "prompts_gender_assumed", 
                           limit: int = 0
                           ) -> List[Dict]:
    """
    Load prompts for a specific model type.
    
    Args:
        model_type: 'base' or 'instruct' (all other types map to 'instruct')
        prompt_file: Name of prompt file
        limit: Optional limit on number of prompts to load (for testing)
    
    Returns:
        List of prompt dictionaries
    """
    prompt_files = {
        'base': data_dir / f"{prompt_file}_base.json",
        'instruct': data_dir / f"{prompt_file}.json"
    }
    
    filepath = prompt_files.get(model_type)
    if not filepath or not filepath.exists():
        raise FileNotFoundError(f"Prompt file for '{model_type}' not found at {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    if limit > 0:
        prompts = prompts[:limit]
    
    return prompts