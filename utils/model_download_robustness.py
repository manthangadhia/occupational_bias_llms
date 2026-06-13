import os
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

models = [
    "swiss-ai/Apertus-8B-2509",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
]

cache_dir = "/cluster/scratch/mgadhia/models/robustness"

for model_name in models:
    print(f"Downloading {model_name}...")
    snapshot_download(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False,  # download if missing
        token=HF_TOKEN
    )