import os
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

models = [
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "microsoft/deberta-v3-large",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
]

cache_dir = "/cluster/scratch/mgadhia/models/bert"

for model_name in models:
    print(f"Downloading {model_name}...")
    snapshot_download(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False,  # download if missing
        token=HF_TOKEN
    )