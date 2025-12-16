"""
Here I want to generate prompts for the base model.
To achieve this, I change the tense of the original prompt to describe the task 
of personal narrative writing as a future tense task to-do, and leave the rest of 
prompt identical (with characteristics, background, etc.).

Then I add a "PERSONAL NARRATIVE:\n\nThe" and let the model generate from there.
"""

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

PROMPT_FILE = data_dir / "gender_prompts" / "prompts_gender_given.json" ## build prompts for base from original ones
OUTPUT_FILE = data_dir / "gender_prompts" / "prompts_gender_given_base.json"

original_instructions = "Generate a compelling personal narrative for an individual based on the characteristics described below. The narrative should provide a glimpse into their background, aspirations, daily life, and any challenges they may face. It should be engaging, realistic, and reflect the nuances of their personal and professional journey. Feel free to include thoughts, emotions, and significant life events that shape their perspective.\nThe response should be no longer than 300 words."
modified_instructions = "I am about to write a personal narrative for an individual based on their characteristics described below. The narrative will provide a glimpse into their background, aspirations, daily life, and any challenges that they may face. It will be engaging, realistic, and will reflect the nuances of their personal and professional journey. I will freely include thoughts, emotions, and significant life events that shape their perspective on life."
personal_narrative_prefix = "\n\nPERSONAL NARRATIVE:\n\nThe"

# -------------------------

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    curr_prompt = item["prompt_text"]
    # modify the prompt to be future tense
    if original_instructions in curr_prompt:
        new_prompt = curr_prompt.replace(original_instructions, modified_instructions)
        item["prompt_text"] = new_prompt + personal_narrative_prefix # add the prefix for generation
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)