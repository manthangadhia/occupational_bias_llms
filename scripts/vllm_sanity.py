from pathlib import Path
import gc
import re
import sys

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
import glob

from utils import load_prompts_for_model, load_model, generate, generate_with_entropy, cleanup_model

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
output_dir = data_dir / "olmo7b_results"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "vllm_sanity_check_results.jsonl"
prompt_dir = data_dir / "robust_prompts"
prompt_dir.mkdir(exist_ok=True)

import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import torch

def main():
    models = {
        "base": [
            "allenai/Olmo-3-1025-7B",
            "mistralai/Ministral-3-8B-Base-2512",
            "swiss-ai/Apertus-8B-2509",  # adjust to exact HF name
        ],
        "instruct": [
            "allenai/Olmo-3-7B-Instruct-SFT",
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "swiss-ai/Apertus-8B-Instruct-2509",
        ]
    }
    temperatures = [0.2, 0.5, 0.7, 1.0]
    num_generations = 50
    with open(output_file, "w") as f_out:
        for category in models:
            print(f"\nRunning inference with {category} models...")
            prompt_files = list(glob(str(prompt_dir / f"*_{category}.json")))

            prompts_df = pd.concat([pd.DataFrame(json.load(open(f))) for f in prompt_files], ignore_index=True)
            prompts = prompts_df["prompt"].tolist()
            
            for model_name in models[category]:
                print(f"\nRunning inference with {model_name} ...")
                if "mistral" in model_name.lower():
                    print(f"Using special handling for Mistral model {model_name} ...")
                    kwargs = {}
                llm = LLM(model=model_name, swap_space=0, max_model_len=4096)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.chat_template:
                    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
                
                outputs = llm.generate(prompts, sampling_params)
            
            # save results before unloading
            save_results(outputs, model_name)
            
            # explicitly free VRAM before next model
            del llm
            gc.collect()
            torch.cuda.empty_cache()