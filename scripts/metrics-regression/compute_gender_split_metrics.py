from pathlib import Path
import argparse
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from utils import save_dataframes

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
results_dir = data_dir / "olmo7b_results"
default_input = results_dir / "olmo7b_temp_results_given.jsonl"


def ensure_nltk_punkt() -> None:
    try:
        word_tokenize("test")
    except LookupError:
        import nltk

        print("NLTK punkt not found. Downloading...")
        nltk.download("punkt")


def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> float:
    if len(texts) < 2:
        return 0.0

    tokenized = [word_tokenize(t.lower()) for t in texts]
    smoothing = SmoothingFunction().method1

    scores_per_n = {n: [] for n in ns}

    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        for n in ns:
            weights = tuple(1 / n for _ in range(n))
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
            scores_per_n[n].append(score)

    mean_scores = [np.mean(scores_per_n[n]) for n in ns]
    return float(np.exp(np.mean(np.log(mean_scores))))


def semantic_diversity(model, texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0

    embeddings = model.encode(texts)
    pairwise_distances = pdist(embeddings, metric="cosine")
    return float(np.mean(pairwise_distances))


def apply_all_metrics(responses_series, semantic_model) -> pd.Series:
    responses_list = [r for r in responses_series if isinstance(r, str) and r.strip()]

    results = {
        "self_bleu": self_bleu(texts=responses_list),
        "semantic_div": semantic_diversity(model=semantic_model, texts=responses_list),
    }

    return pd.Series(results)


def read_results_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def expand_entropy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "entropy_analysis" not in df.columns:
        return df

    entropy_df = pd.json_normalize(df["entropy_analysis"])
    for col in entropy_df.columns:
        if col in df.columns:
            df[col] = df[col].fillna(entropy_df[col])
        else:
            df[col] = entropy_df[col]

    return df.drop(columns=["entropy_analysis"])


def normalize_gender_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("none").astype(str).str.strip().str.lower()
    replacements = {
        "unspecified": "none",
        "none": "none",
        "nan": "none",
        "": "none",
    }
    return normalized.replace(replacements)


def compute_occupation_gender_model_metrics_summary(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    required_cols = {"occupation", "gender", "model_key"}
    if not required_cols.issubset(summary_df.columns):
        return pd.DataFrame()
    if "gender" not in raw_df.columns or "model_key" not in raw_df.columns:
        return pd.DataFrame()

    available_metrics = [m for m in metric_cols if m in summary_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    metrics_df = summary_df.copy()
    metrics_df["_gender"] = normalize_gender_series(metrics_df["gender"])
    metrics_df = metrics_df[metrics_df["_gender"].isin(["male", "female"])]
    if metrics_df.empty:
        return pd.DataFrame()

    grouped = metrics_df.groupby(["occupation", "_gender", "model_key"], dropna=False)
    agg_df = grouped[available_metrics].agg(["mean", "std", "median", "count"]).reset_index()

    flat_columns = []
    for col in agg_df.columns:
        if col in {"occupation", "_gender", "model_key"}:
            flat_columns.append("gender" if col == "_gender" else col)
        elif isinstance(col, tuple):
            metric_name = col[0]
            stat_name = col[1] if len(col) > 1 else ""
            flat_columns.append(f"{metric_name}_{stat_name}".rstrip("_"))
        else:
            flat_columns.append(str(col))

    agg_df.columns = flat_columns
    if "_gender" in agg_df.columns and "gender" not in agg_df.columns:
        agg_df = agg_df.rename(columns={"_gender": "gender"})

    raw_gender_df = raw_df.copy()
    raw_gender_df["_gender"] = normalize_gender_series(raw_gender_df["gender"])
    raw_gender_df = raw_gender_df[raw_gender_df["_gender"].isin(["male", "female"])]
    if raw_gender_df.empty:
        return agg_df

    response_counts = (
        raw_gender_df.groupby(["occupation", "_gender", "model_key"], dropna=False)
        .size()
        .reset_index(name="response_count")
        .rename(columns={"_gender": "gender"})
    )

    return agg_df.merge(response_counts, on=["occupation", "gender", "model_key"], how="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(default_input),
        help=f"Path to a results file (json or jsonl). Default: {default_input}",
    )
    parser.add_argument(
        "--output-prefix",
        default="given_occupation_gender_metrics_",
        help="Prefix for output JSON filename in results_dir",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = read_results_file(input_path)
    df = expand_entropy_columns(df)

    group_cols = [
        col
        for col in ["model_key", "profile_id", "temperature", "prompt_case"]
        if col in df.columns
    ]
    if not group_cols:
        raise ValueError("No group columns found (expected model_key, profile_id, temperature, prompt_case)")

    meta_candidates = [
        "model_name",
        "occupation",
        "occupation_category",
        "attended_university",
        "gender",
    ]
    meta_cols = [col for col in meta_candidates if col in df.columns and col not in group_cols]

    entropy_candidates = [
        "mean_entropy",
        "max_entropy",
        "min_entropy",
        "std_entropy",
        "mean_entropy_nucleus",
    ]
    entropy_cols = [col for col in entropy_candidates if col in df.columns]

    meta_summary = (
        df.groupby(group_cols, dropna=False)[meta_cols].first().reset_index()
        if meta_cols
        else df[group_cols].drop_duplicates()
    )

    entropy_summary = None
    if entropy_cols:
        entropy_summary = (
            df.groupby(group_cols, dropna=False)[entropy_cols]
            .mean()
            .reset_index()
            .rename(columns={col: f"avg_{col}" for col in entropy_cols})
        )

    ensure_nltk_punkt()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    metrics_aggregator = partial(
        apply_all_metrics,
        semantic_model=semantic_model,
    )

    tqdm.pandas(desc="Applying metrics by group")
    metrics_long = (
        df.groupby(group_cols, dropna=False)["response"]
        .progress_apply(metrics_aggregator)
        .reset_index()
    )

    metric_name_col = None
    for candidate in ["level_4", "level_3", "level_2", "level_1"]:
        if candidate in metrics_long.columns:
            metric_name_col = candidate
            break

    if metric_name_col and "response" in metrics_long.columns:
        metrics_wide = (
            metrics_long.pivot_table(
                index=group_cols,
                columns=metric_name_col,
                values="response",
                aggfunc="first",
            )
            .reset_index()
        )
        metrics_wide.columns.name = None
    else:
        metrics_wide = metrics_long

    summary_df = meta_summary.merge(metrics_wide, on=group_cols, how="left")
    if entropy_summary is not None:
        summary_df = summary_df.merge(entropy_summary, on=group_cols, how="left")

    metric_candidates = ["self_bleu", "semantic_div", "avg_mean_entropy"]
    gender_metrics_df = compute_occupation_gender_model_metrics_summary(
        summary_df,
        df,
        metric_candidates,
    )

    if gender_metrics_df.empty:
        raise ValueError("Gender-split occupation metrics are empty. Check gender/occupation columns.")

    output_key = f"{args.output_prefix}{input_path.stem}"
    save_dataframes({output_key: gender_metrics_df}, results_dir)
    print(f"Saved: {output_key}.json in {results_dir}")
