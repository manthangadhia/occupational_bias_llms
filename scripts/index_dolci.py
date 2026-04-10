print("Creating index for Dolci-SFT samples that are relevant given gender_assumed and gender_given prompts.")
# Setup paths
import os
from pathlib import Path
import sys

cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < scripts < occ_bias < root > models         # structure on euler
dolci_data_dir = root_dir / "occ_bias" / "data" / "dolci_sft"
dolci_data_path = root_dir / "occ_bias" / "data" / "dolci_sft" / "dolci_sft.parquet"  # path to parquet file on euler

# Add utils to path
sys.path.insert(0, str(root_dir))
from occ_bias.utils import load_json_data, load_embedding_model, embed_batch

staged_models = os.getenv("HF_HOME") or os.getenv("MODEL_ROOT")
models_dir = Path(staged_models) if staged_models else (root_dir / "models" / "embeddings")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Model cache directory: {models_dir}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import pandas as pd
import faiss
import numpy as np

EMBED_BATCH_SIZE = 64
FLUSH_EVERY = 10_000
EMBEDDING_DIM = 1024
MAX_LENGTH = 500

import json
import argparse
from tqdm import tqdm

# Helper Functions

def load_gender_prompts(data_dir: Path) -> pd.DataFrame:
    """
    Load the gender_assumed and gender_given prompts from the data directory
    (instruct-prompts only) and return as a DataFrame
    """
    return load_json_data(data_dir, prefix="prompts_gender", exclude_keyword="base")

def count_tokens(text, tokenizer): 
    """Fast token counting without padding or tensors"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_and_concatenate(text, tokenizer, max_length=MAX_LENGTH):
    """For prompts that exceed max length, truncate max_length/2 at the beginning and end and concatenate."""
    half_max = max_length // 2
    truncated_start = tokenizer.decode(tokenizer.encode(text)[:half_max], skip_special_tokens=True)
    truncated_end = tokenizer.decode(tokenizer.encode(text)[-half_max:], skip_special_tokens=True)
    return truncated_start + " " + truncated_end

# Main functionality
def main(args):
    # Load model and tokenizer as object in retrieval_utils
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = load_embedding_model(device=device, cache_dir=models_dir)
    tokenizer = model.tokenizer

    # Load the dataset from local memory (parquet file on euler)
    ds = load_dataset("parquet", data_files=str(dolci_data_path))["train"]
    total_len_ds = len(ds)
    # limit the domains that are loaded and processed
    allowed_domains = {"chat", "other", "reasoning", "safety", "precise if"}
    ds = ds.filter(lambda x: x["domain"] and x["domain"].strip().lower() in allowed_domains)
    print(f"Dataset loaded with {len(ds)} samples after filtering to allowed domains. {len(ds)/total_len_ds:.2%} of total samples retained.")

    # Set up faiss index and metadata buffers and flushing logic
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata, text_buffer, meta_buffer = [], [], []
    total_vectors_added = 0

    def flush_buffer(text_buffer, meta_buffer, model, index, metadata):
        embs = embed_batch(text_buffer, model, prefix="passage: ")
        index.add(embs)
        metadata.extend(meta_buffer)
        text_buffer.clear()
        meta_buffer.clear()

    # Loop through dataset, keep count, embed, and regularly flush to faiss index
    for hf_row_idx, row in enumerate(tqdm(ds)):
        for turn_idx, turn in enumerate(row["messages"]):
            if turn["role"] != "user":
                continue
            text = turn["content"].strip()
            if not text:
                continue
            
            is_truncated = False
            token_len = count_tokens(text, tokenizer)
            faiss_id = total_vectors_added + len(text_buffer)

            # If sequence is too long (as some of them are), then keep relevant info from beginning and end
            if token_len > MAX_LENGTH:
                text = truncate_and_concatenate(text, tokenizer, max_length=MAX_LENGTH)
                is_truncated = True

            meta_buffer.append({
                "faiss_id": faiss_id,
                "hf_row_idx": hf_row_idx,
                "turn_idx": turn_idx,
                "role": "user",
                "preview": text[:200],
                "token_len": token_len,
                "is_truncated": is_truncated,
            })
            text_buffer.append(text)

            if len(text_buffer) >= FLUSH_EVERY:
                flush_buffer(text_buffer, meta_buffer, model, index, metadata)
                total_vectors_added = index.ntotal  # let FAISS be the source of truth

    if text_buffer:
        flush_buffer(text_buffer, meta_buffer, model, index, metadata)

    faiss.write_index(index, str(dolci_data_dir / "dolci_user.index"))
    pd.DataFrame(metadata).to_parquet(dolci_data_dir / "dolci_meta.parquet", index=False)
    print(f"Done. {index.ntotal} vectors indexed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None)
    main(parser.parse_args())