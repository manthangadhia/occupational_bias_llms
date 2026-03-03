from pathlib import Path
import pandas as pd
import sys
import argparse
from functools import partial

import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from utils import load_json_data, save_dataframes

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
results_dir = data_dir / "olmo7b_results"


def perplexity(model, tokenizer, texts: list[str]) -> float:
    """
    For a given set of texts, compute per-text perplexity using a causal LM (e.g. GPT-2).
    Returns average perplexity across all texts.
    """
    model.eval()
    perplexities = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            perplexities.append(torch.exp(loss).item())

    return float(np.mean(perplexities))


def apply_perplexity(responses_series, perplexity_model, perplexity_tokenizer):
    responses_list = list(responses_series)
    results = {
        "perplexity": perplexity(
            model=perplexity_model,
            tokenizer=perplexity_tokenizer,
            texts=responses_list,
        ),
    }
    return pd.Series(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="given", help="Keyword to filter the files to be loaded")
    args = parser.parse_args()

    data_frames = load_json_data(results_dir, file_name_keyword=args.keyword)
    perplexity_dfs = {}
    tqdm.pandas(desc="Applying perplexity by group")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")
    perplexity_model.to(device)

    perplexity_aggregator = partial(
        apply_perplexity,
        perplexity_model=perplexity_model,
        perplexity_tokenizer=perplexity_tokenizer,
    )

    cols_to_keep = [
        "model_key",
        "profile_id",
        "temperature",
        "occupation_category",
        "attended_university",
        "response_number",
        "mean_entropy",
        "mean_entropy_nucleus",
        "gender",
    ]

    for k, df in tqdm(data_frames.items(), total=len(data_frames), desc="Processing files"):
        perplexity_df = df[cols_to_keep].copy()
        metrics = (
            df.groupby(["profile_id", "temperature"])["response"]
            .progress_apply(perplexity_aggregator)
            .unstack()
            .reset_index()
        )

        perplexity_df = perplexity_df.merge(
            metrics,
            on=["profile_id", "temperature"],
            how="left",
        )

        output_key = "perplexity_" + k
        perplexity_dfs[output_key] = perplexity_df

    save_dataframes(perplexity_dfs, results_dir)
