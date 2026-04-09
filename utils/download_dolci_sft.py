"""This script downloads the Dolci Instruct SFT dataset from Hugging Face and saves it as a parquet file in the data directory."""

from datasets import load_dataset
import sys
import os
from pathlib import Path

cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < utils < occ_bias < root > models
data_dir = root_dir / "occ_bias" / "data"
dolci_sft_dir = data_dir / "dolci_sft"
dolci_sft_dir.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    ds = load_dataset(
        "allenai/Dolci-Instruct-SFT",
        split="train",
        token=HF_TOKEN
    )

    ds.to_parquet(dolci_sft_dir / "dolci_sft.parquet")

if __name__ == "__main__":
    main()