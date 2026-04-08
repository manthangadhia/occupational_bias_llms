import os
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

models = [
    "meta-llama/Llama-3.1-8B"
]

cache_dir = "/cluster/scratch/mgadhia/models/llama"

for model_name in models:
    print(f"Downloading {model_name}...")
    snapshot_download(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False,  # download if missing
        token=HF_TOKEN
    )