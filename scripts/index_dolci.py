print("Creating index for Dolci-SFT samples that are relevant given gender_assumed and gender_given prompts.")
# Setup paths
import os
from pathlib import Path
import sys

cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < scripts < occ_bias < root > models         # structure on euler
gender_prompts_dir = root_dir / "occ_bias" / "data" / "gender_prompts"
dolci_data_dir = root_dir / "occ_bias" / "data" / "dolci_sft"
dolci_data_path = root_dir / "occ_bias" / "data" / "dolci_sft" / "dolci_sft.parquet"  # path to parquet file on euler
dolci_index_path = dolci_data_dir / "dolci_user.index"
dolci_meta_path = dolci_data_dir / "dolci_meta.parquet"

# Add utils to path
sys.path.insert(0, str(root_dir))
from occ_bias.utils import load_json_data, load_embedding_model, embed_batch, query_embedding_model

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
import datasets
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
    return load_json_data(data_dir, file_name_keyword="assumed", exclude_keyword="base")

def load_dolci_data(data_path: Path) -> datasets.Dataset:
    """Load the Dolci-SFT dataset from the given parquet file path."""
    return load_dataset("parquet", data_files=str(data_path))["train"]

def count_tokens(text, tokenizer): 
    """Fast token counting without padding or tensors"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_and_concatenate(text, tokenizer, max_length=MAX_LENGTH):
    """For prompts that exceed max length, truncate max_length/2 at the beginning and end and concatenate."""
    half_max = max_length // 2
    truncated_start = tokenizer.decode(tokenizer.encode(text)[:half_max], skip_special_tokens=True)
    truncated_end = tokenizer.decode(tokenizer.encode(text)[-half_max:], skip_special_tokens=True)
    return truncated_start + " " + truncated_end

def process_prompt_text(prompt_text, tokenizer, max_length=MAX_LENGTH):
    """Ensure prompt text is within max token length, else truncate and concatenate."""
    token_len = count_tokens(prompt_text, tokenizer)
    if token_len > max_length:
        return truncate_and_concatenate(prompt_text, tokenizer, max_length)
    return prompt_text

def retrieve_text_for_retrieved_samples(df_samples, ds):
    # Explode df_samples so that each row corresponds to one retrieved sample
    exploded = df_samples.explode("retrieved_samples").reset_index(drop=True)
    retrieved_texts = []
    for idx, row in exploded.iterrows():
        retrieved_sample = row["retrieved_samples"]
        hf_row_idx = retrieved_sample["hf_row_idx"]
        turn_idx = retrieved_sample["turn_idx"]
        text = ds[hf_row_idx]["messages"][turn_idx]["content"]
        retrieved_texts.append(text)
    exploded["retrieved_text"] = retrieved_texts
    return exploded

# Main functionality
def create_index(model, tokenizer):
    # Load the dataset from local memory (parquet file on euler)
    ds = load_dolci_data(dolci_data_path)
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

    faiss.write_index(index, str(dolci_index_path))
    pd.DataFrame(metadata).to_parquet(dolci_meta_path, index=False)
    print(f"Done. {index.ntotal} vectors indexed.")

def query_index(model, tokenizer, top_k = 10):
    # Load prompts
    prompts_dfs = load_gender_prompts(gender_prompts_dir)
    print(f"Loaded {len(prompts_dfs)} gender prompt files.")
    if len(prompts_dfs) == 1: # which it should be, then just take the one dataframe out of the dict
        df = next(iter(prompts_dfs.values()))
    else:        raise ValueError(f"Expected one prompts dataframe, but got {len(prompts_dfs)}.")

    # Ensure each prompt is within the max token length, else process
    df["prompt_text"] = df["prompt_text"].apply(lambda x: process_prompt_text(x, tokenizer))
    # Pass final prompt_text as query_text
    index = faiss.read_index(str(dolci_index_path))
    results = df["prompt_text"].apply(
        lambda x: query_embedding_model(x, model, index, k=top_k)
    )
    df["sample_distances"], df["sample_faiss_ids"] = zip(*results)    # Use the faiss_ids to get hf_row_idx and turn_idx from the metadata, then get corresponding text from dolci
    meta_df = pd.read_parquet(dolci_meta_path)
    df["retrieved_samples"] = df["sample_faiss_ids"].apply(lambda faiss_ids: meta_df[meta_df["faiss_id"].isin(faiss_ids)][["hf_row_idx", "turn_idx"]].to_dict(orient="records"))
    #TODO: for each retrieved sample, also get the corresponding text from the original dataset and include in the final dataframe
    ds = load_dolci_data(dolci_data_path)
    df_with_text = retrieve_text_for_retrieved_samples(df, ds)

    # # Save the resulting dataframe with prompts and retrieved samples to a new parquet file for analysis
    output_path = gender_prompts_dir / "gender_assumed_prompts_with_retrieved_dolci.parquet"
    df.to_parquet(output_path, index=False)

    text_output_path = gender_prompts_dir / "gender_assumed_prompts_with_retrieved_dolci_text.parquet"
    df_with_text.to_parquet(text_output_path, index=False)
    print(f"Querying complete. Results saved to {output_path} and {text_output_path}.")

def main(args):
    # Load model and tokenizer as object in retrieval_utils
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = load_embedding_model(device=device, cache_dir=models_dir)
    tokenizer = model.tokenizer

    if args.pipeline == "index":
        print("Creating index...")
        create_index(model, tokenizer)
    else: # args.pipeline == "query"
        print("Querying...")
        query_index(model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--pipeline", type=str, default="index", help="Whether to run the indexing or querying pipeline.")
    main(parser.parse_args())