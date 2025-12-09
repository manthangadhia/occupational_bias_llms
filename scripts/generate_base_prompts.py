# Here I would take the content of each prompt profile from gender_given.json
# and generate a prompt for a base model by:
# base_prompt = [ift_prompt] + [first 50characters of response from ift model]

import json
import sys
from pathlib import Path

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# -------------------------
# configuration
# -------------------------
data_dir = root_dir / "data"

PROMPT_FILE = data_dir / "output_ift.jsonl" ## build prompts for base from ift outputs
OUTPUT_FILE = data_dir / "gender_prompts" / "prompts_ggiven_base_200.json"

# -------------------------

def create_prompt_with_narrative(prompt, narrative, chars=100):
    """Create a new prompt by combining the original prompt with narrative snippet."""
    narrative_snippet = narrative[:chars]
    combined_prompt = f"{prompt}\n{narrative_snippet}"
    return combined_prompt

def main():
    # Load prompts from IFT output JSON file
    prompts_with_narratives = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompts_with_narratives.append(entry)
    
    print(f"Loaded {len(prompts_with_narratives)} prompts from {PROMPT_FILE}")

    # Create new prompts for base model
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        all_new_prompts = []
        for entry in prompts_with_narratives:
            profile_id = entry["profile_id"]
            prompt = entry["prompt"]
            narrative = entry["response"]
            
            new_prompt = create_prompt_with_narrative(prompt, narrative, chars=200)
            
            new_entry = {
                "profile_id": profile_id,
                "prompt_text": new_prompt
            }
            all_new_prompts.append(new_entry)
        json.dump(all_new_prompts, out, indent=2)

if __name__ == "__main__":
    main()