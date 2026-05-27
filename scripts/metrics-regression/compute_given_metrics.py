from pathlib import Path
import sys
import json
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist
from scipy import stats
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Add utils to path
root_dir = Path(__file__).parent.parent.parent
print(f"Root directory: {root_dir}")
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


def zscore_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = values.mean()
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.where(values.notna(), 0.0, np.nan), index=values.index)
    return (values - mean) / std


def boxcox_if_skewed(
    series: pd.Series,
    skew_threshold: float = 0.25,
) -> tuple[pd.Series, dict]:
    cleaned = series.dropna().astype(float)
    skew_value = float(cleaned.skew()) if not cleaned.empty else float("nan")
    info = {
        "transform": "none",
        "skew": skew_value,
        "skew_threshold": skew_threshold,
        "lambda": None,
        "shift": 0.0,
    }

    if cleaned.empty or cleaned.nunique() < 2:
        return series, info

    if not np.isfinite(skew_value) or abs(skew_value) < skew_threshold:
        return series, info

    shift = 0.0
    min_val = cleaned.min()
    if min_val <= 0:
        shift = 1 - min_val

    transformed, lambda_value = stats.boxcox(cleaned + shift)
    transformed_series = series.copy()
    transformed_series.loc[cleaned.index] = transformed

    info.update(
        {
            "transform": "boxcox",
            "lambda": float(lambda_value),
            "shift": float(shift),
        }
    )

    return transformed_series, info


def build_model_indicators(model_series: pd.Series) -> pd.DataFrame:
    normalized = model_series.fillna("").astype(str).str.strip().str.lower()
    indicators = pd.DataFrame(index=model_series.index)
    indicators["is_sft"] = normalized.str.contains("sft").astype(int)
    indicators["is_dpo"] = normalized.str.contains("dpo").astype(int)
    indicators["is_rlvr"] = normalized.str.contains("rlvr").astype(int)
    return indicators


if __name__ == "__main__":
    input_path = default_input
    # sample input from file: {"model_key": "base", "model_name": "allenai/Olmo-3-1025-7B", "prompt_case": "given", "profile_id": 1, "temperature": 0.2, "gender": "female", "occupation": "carpenter", "attended_university": "yes", "response_number": 9, "response": "In the heart of a bustling city, there lived a woman named Elara, whose life was a testament to resilience and passion. Elara was a carpenter, a profession she...", "entropy_analysis": {"mean_entropy": 0.21111416280865106, "max_entropy": 1.4453125, "min_entropy": 2.8525210332791787e-27, "std_entropy": 0.315127907344944}}
    df = read_results_file(input_path)
    df = expand_entropy_columns(df)

    ensure_nltk_punkt()

    df["temperature"] = pd.to_numeric(df.get("temperature"), errors="coerce")
    df = df.dropna(subset=["temperature", "response"])

    group_keys = ["profile_id", "temperature", "model_key"]
    meta_cols = [
        "model_key",
        "model_name",
        "prompt_case",
        "profile_id",
        "temperature",
        "gender",
        "occupation",
        "attended_university",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]
    meta_value_cols = [col for col in meta_cols if col not in group_keys]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    tqdm.pandas(desc="Applying metrics by group")
    metrics_aggregator = partial(apply_all_metrics, semantic_model=semantic_model)

    metrics_df = (
        df.groupby(group_keys, dropna=False)["response"]
        .progress_apply(metrics_aggregator)
        .unstack()
        .reset_index()
    )

    entropy_cols = [
        col
        for col in ["mean_entropy", "mean_entropy_nucleus"]
        if col in df.columns
    ]
    entropy_df = pd.DataFrame()
    if entropy_cols:
        entropy_df = (
            df.groupby(group_keys, dropna=False)[entropy_cols]
            .mean()
            .reset_index()
        )

    response_counts = (
        df.groupby(group_keys, dropna=False)
        .size()
        .reset_index(name="response_count")
    )

    if meta_value_cols:
        meta_df = df.groupby(group_keys, dropna=False)[meta_value_cols].first().reset_index()
    else:
        meta_df = df[group_keys].drop_duplicates().reset_index(drop=True)

    summary_df = meta_df.merge(metrics_df, on=group_keys, how="left")
    if not entropy_df.empty:
        summary_df = summary_df.merge(entropy_df, on=group_keys, how="left")
    summary_df = summary_df.merge(response_counts, on=group_keys, how="left")

    metric_cols = ["self_bleu", "semantic_div"]
    for col in entropy_cols:
        if col not in metric_cols:
            metric_cols.append(col)

    regression_df = summary_df.copy()
    regression_df["gender_norm"] = normalize_gender_series(regression_df.get("gender"))
    regression_df["is_female"] = (regression_df["gender_norm"] == "female").astype(int)

    model_indicators = build_model_indicators(regression_df.get("model_key"))
    regression_df = pd.concat([regression_df, model_indicators], axis=1)
    regression_df["temperature_z"] = zscore_series(regression_df["temperature"])

    transform_metadata = []
    regression_outputs = []
    coeff_rows = []

    formula_template = (
        "{metric}_z ~ is_female + is_sft + is_dpo + is_rlvr "
        "+ is_female:is_sft + is_female:is_dpo + is_female:is_rlvr + temperature_z"
    )

    for metric in metric_cols:
        if metric not in regression_df.columns:
            continue

        transformed, info = boxcox_if_skewed(regression_df[metric])
        metric_z = zscore_series(transformed)
        metric_z_name = f"{metric}_z"
        regression_df[metric_z_name] = metric_z

        info.update({"metric": metric})
        transform_metadata.append(info)

        formula = formula_template.format(metric=metric)
        model_data = regression_df.dropna(subset=[metric_z_name])
        if model_data.empty:
            continue

        model = smf.ols(formula, data=model_data).fit()
        regression_outputs.append(
            {
                "metric": metric,
                "nobs": int(model.nobs),
                "r2": float(model.rsquared),
                "r2_adj": float(model.rsquared_adj),
                "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else float("nan"),
            }
        )

        conf_int = model.conf_int()
        for term, coef in model.params.items():
            conf_low, conf_high = conf_int.loc[term].tolist()
            coeff_rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "coef": float(coef),
                    "stderr": float(model.bse[term]),
                    "pvalue": float(model.pvalues[term]),
                    "conf_low": float(conf_low),
                    "conf_high": float(conf_high),
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                }
            )

    output_dir = results_dir / "metrics_given"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "given_metrics_summary": summary_df,
        "given_metrics_regression_input": regression_df,
        "given_metrics_regression_coeffs": pd.DataFrame(coeff_rows),
        "given_metrics_regression_models": pd.DataFrame(regression_outputs),
        "given_metrics_transform_metadata": pd.DataFrame(transform_metadata),
    }

    save_dataframes(outputs, output_dir)
    for key, df_out in outputs.items():
        df_out.to_csv(output_dir / f"{key}.csv", index=False)

    metadata = {
        "input_path": str(input_path),
        "group_keys": group_keys,
        "metric_cols": metric_cols,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "semantic_model": "all-MiniLM-L6-v2",
    }
    with (output_dir / "given_metrics_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)