"""This script downloads the Dolci Instruct SFT dataset from Hugging Face and saves it as a parquet file in the data directory."""

from datasets import load_dataset
import pandas as pd
import json
import sys
import os
from pathlib import Path
import gc

cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < utils < occ_bias < root > models
data_dir = root_dir / "occ_bias" / "data"
dolci_sft_dir = data_dir / "dolci_sft"
dolci_sft_dir.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import argparse
def arg_to_bool(value: int):
    return bool(value)

def verify_proper_save(file_path: Path):
    """Verify that the parquet file was saved properly by trying to read it back."""
    try:
        df = pd.read_parquet(file_path, engine="pyarrow")
        print("------ Verifying the saved parquet file ------")
        print(f"Successfully read the parquet file with {len(df)} rows and columns: {df.columns.tolist()}")
        print(f"The first row of the dataframe is: {df.iloc[0]}")
    except Exception as e:
        print(f"Error reading the parquet file: {e}")
        sys.exit(1)

def main():
    ds = load_dataset(
        "allenai/Dolci-Instruct-SFT",
        split="train",
        token=HF_TOKEN
    )
    ds.to_parquet(dolci_sft_dir / "dolci_sft.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Dolci Instruct SFT dataset")
    parser.add_argument("--verify-only", default=1, type=int, help="Only verify the saved parquet file without downloading")
    args = parser.parse_args()
    args.verify_only = arg_to_bool(args.verify_only)
    print(f"Arguments: {args}")
    if arg_to_bool(args.verify_only):
        verify_proper_save(dolci_sft_dir / "dolci_sft.parquet")
    else:
        main()    
        gc.collect()   # clean up memory after downloading and saving the dataset
        verify_proper_save(dolci_sft_dir / "dolci_sft.parquet")