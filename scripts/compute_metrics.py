from pathlib import Path
import pandas as pd
import sys
import argparse
from functools import partial

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from utils import load_json_data, save_dataframes
# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
results_dir = data_dir / "olmo7b_results"

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm.auto import tqdm

def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> dict:
    """
    For a given set of texts, compute self-bleu across all pairs using nltk.sentence_bleu.
    For each text, treat it as hypothesis and the rest as references.
    
    Returns combined geometric mean over n-gram orders, I don't think I need the per n-gram results.
    """
    tokenized = [word_tokenize(t.lower()) for t in texts]
    # smoothing = SmoothingFunction().method1                 # This is useful when texts being compared are short, mine are quite lengthy

    scores_per_n = {n: [] for n in ns}

    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        for n in ns:
            weights = tuple(1/n for _ in range(n))
            score = sentence_bleu(refs, hyp, weights=weights) # can add smoothing function if needed
            scores_per_n[n].append(score)

    results = {f"self_bleu_{n}": float(np.mean(scores_per_n[n])) for n in ns}
    
    # geometric mean across n-gram orders
    mean_scores = [np.mean(scores_per_n[n]) for n in ns]
    results["self_bleu_combined"] = float(np.exp(np.mean(np.log(mean_scores))))

    return results["self_bleu_combined"]

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
# semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_diversity(model, texts: list[str]):
    """
    For a given set of texts, compute their pairwise semantic similarity using MiniLM-v6.
    Return average pairwise cosine distance.
    """
    if len(texts) < 2:
        return 0.0
    
    embeddings = model.encode(texts)
    pairwise_distances = pdist(embeddings, metric='cosine')
    avg_distance = np.mean(pairwise_distances)
    
    return avg_distance

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")

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

def apply_all_metrics(responses_series, semantic_model, perplexity_model, perplexity_tokenizer):
    responses_list = list(responses_series)                     # convert the grouped series to a list

    results = {
        "self_bleu": self_bleu(texts=responses_list),
        "semantic_div": semantic_diversity(model=semantic_model, 
                                           texts=responses_list),
        "perplexity": perplexity(model=perplexity_model, 
                                 tokenizer=perplexity_tokenizer, 
                                 texts=responses_list),
    }

    return pd.Series(results)

def load_test():
    filename = "sft_test.json"
    filepath = results_dir / filename
    df = pd.read_json(filepath)

    return {"sft_test": df}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="given", help="Keyword to filter the files to be loaded")
    args = parser.parse_args()    
    
    # Load data
    data_frames = load_json_data(results_dir, file_name_keyword=args.keyword)
    # data_frames = load_test()
    metric_dfs = {}
    tqdm.pandas(desc="Applying metrics by group")

    # Initialize shared models once and reuse for all groups/dataframes
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")
    perplexity_model.to(device)

    metrics_aggregator = partial(
        apply_all_metrics,
        semantic_model=semantic_model,
        perplexity_model=perplexity_model,
        perplexity_tokenizer=perplexity_tokenizer,
    )

    # go through one df at a time, 
    cols_to_keep = ['model_key', 'profile_id', 'temperature',
       'occupation_category', 'attended_university', 'response_number',
       'mean_entropy', 'mean_entropy_nucleus', 'gender']
    for k, df in tqdm(data_frames.items(), total=len(data_frames), desc="Processing files"):
        metric_df = df[cols_to_keep].copy()
        # group by profile id and temp
        metrics = (
            # compute metrics for each profile id and temp
            df.groupby(["profile_id", "temperature"])["response"]
            .progress_apply(metrics_aggregator)
            .unstack()
            .reset_index()
        )
        # store results in new df in a new dict
        metric_df = metric_df.merge(
            metrics,
            on=["profile_id", "temperature"],
            how="left"
        )
        metric_k = "metrics_" + k
        metric_dfs[metric_k] = metric_df
    
    # save dict to json using util function
    save_dataframes(metric_dfs, results_dir)