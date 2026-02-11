# utils/prompt_loader.py
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
prompts_dir = data_dir / "gender_prompts"


def get_prompt_files(prompts_dir: Path = prompts_dir) -> Dict:
    """
    Function to get all the paths with clear descriptive labels for all prompt files

    Args:
        prompts_dir: Path to directory where prompts are stored 
    
    Returns: 
        Dict containing a name for each prompt_file and the path to that file
    """

    given_prompts = prompts_dir / "prompts_gender_given_detailed.json"
    given_prompts_base = prompts_dir / "prompts_gender_given_detailed_base.json"
    assumed_prompts = prompts_dir / "prompts_gender_assumed_detailed.json"    
    assumed_prompts_base = prompts_dir / "prompts_gender_assumed_detailed_base.json"

    all_prompt_files = {
        "given": given_prompts,
        "given_base": given_prompts_base,
        "assumed": assumed_prompts,
        "assumed_base": assumed_prompts_base
    }

    return all_prompt_files

def load_prompts_for_model(model_type: str, 
                           prompt_case: str,
                           all_prompt_files: dict = get_prompt_files(), 
                           limit: int = 0
                           ) -> List[Dict]:
    """
    Load the correct prompt file based on filename and model_type.
    
    Args:
        model_type: 'base' or 'instruct' (all other types map to 'instruct')
        prompt_case: 'given' or 'assumed' to indicate which class of prompts to load
        all_prompt_files: a dict of all 4 available prompt files with names and partial paths
        limit: Optional limit on number of prompts to load (for testing)
    
    Returns:
        Pandas DF containing prompt text along with all other detailed info stored in the json
    """
    # prompt_files = {
    #     'base': data_dir / f"{prompt_file}_base.json",
    #     'instruct': data_dir / f"{prompt_file}.json"
    # }

    # I want a prompt key which is either given/given_base/assumed/assumed_base
    prompt_key = f"{prompt_case}_base" if model_type == 'base' else prompt_case
    
    filepath = all_prompt_files.get(prompt_key)
    if not filepath or not filepath.exists():
        raise FileNotFoundError(f"Prompt file for '{model_type} model' and '{prompt_case} prompt' not found at {filepath}")
    
    prompts_df = pd.read_json(filepath)
    
    if limit > 0:
        prompts_df = prompts_df[:limit]
    
    return prompts_df