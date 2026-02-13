import os
from huggingface_hub import snapshot_download

models = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct-DPO", 
    "allenai/Olmo-3-7B-Instruct-SFT"
]

cache_dir = "/cluster/scratch/mg/olmo/models"

for model_name in models:
    print(f"Downloading {model_name}...")
    snapshot_download(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False  # download if missing
    )