from pathlib import Path
import pandas as pd
import sys
import argparse
from functools import partial
import torch

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from utils import save_dataframes
# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
results_dir = data_dir / "olmo7b_results"
assumed_results = results_dir / "olmo7b_temp_results_assumed.jsonl"

"""
Example output format for assumed_results:
{"model_key": "sft", "model_name": "allenai/Olmo-3-7B-Instruct-SFT", "prompt_case": "assumed", "profile_id": 28, "temperature": 0.7, "occupation": "consultant", "attended_university": "yes", "response_number": 3, "response": " You are allowed to make assumptions about the person's personality, based on the provided characteristics. The occupation title should be mentioned in the response.\n\nassistant\nGrowing up in a bustling coastal town, I was always fascinated by the constant ebb and flow of ideas...", "entropy_analysis": {"mean_entropy": 0.9206281426341836, "max_entropy": 3.046875, "min_entropy": 1.3096723705530167e-10, "std_entropy": 0.7879512841699567}}
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm.auto import tqdm

def ensure_nltk_punkt() -> None:
    try:
        word_tokenize("test")
    except LookupError:
        import nltk

        print("NLTK punkt not found. Downloading...")
        nltk.download("punkt")

def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> float:
    """
    For a given set of texts, compute self-bleu across all pairs using nltk.sentence_bleu.
    For each text, treat it as hypothesis and the rest as references.

    Returns combined geometric mean over n-gram orders.
    """
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

    # geometric mean across n-gram orders
    mean_scores = [np.mean(scores_per_n[n]) for n in ns]
    return float(np.exp(np.mean(np.log(mean_scores))))

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
# semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_diversity(model, texts: list[str]) -> float:
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


def load_results_files(
    results_path: Path,
    file_name_keyword: str | None = None,
    input_path: Path | None = None,
) -> dict:
    data_frames = {}

    derived_prefixes = (
        "metrics_summary_",
        "occupation_summary_",
        "occupation_diff_",
        "occupation_metrics_",
        "occupation_model_metrics_",
        "occupation_delta_metrics_",
        "gender_summary_",
    )

    def should_skip(file: Path) -> bool:
        return file.stem.startswith(derived_prefixes)

    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        data_frames[input_path.stem] = read_results_file(input_path)
        return data_frames

    for file in results_path.glob("*.json"):
        if should_skip(file):
            continue
        if file_name_keyword and file_name_keyword not in file.name:
            continue
        data_frames[file.stem] = read_results_file(file)

    for file in results_path.glob("*.jsonl"):
        if should_skip(file):
            continue
        if file_name_keyword and file_name_keyword not in file.name:
            continue
        data_frames[file.stem] = read_results_file(file)

    return data_frames

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

def plot_metrics(summary_df: pd.DataFrame, output_dir: Path, metric_cols: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots (matplotlib unavailable): {exc}")
        return

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    has_temp = "temperature" in summary_df.columns
    has_model = "model_key" in summary_df.columns

    for metric in metric_cols:
        if metric not in summary_df.columns:
            print(f"Skipping plot for missing metric: {metric}")
            continue
        metric_df = summary_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue

        if has_temp and has_model:
            fig, ax = plt.subplots(figsize=(8, 5))
            if sns:
                sns.boxplot(data=metric_df, x="temperature", y=metric, hue="model_key", ax=ax)
            else:
                metric_df.boxplot(column=metric, by=["temperature", "model_key"], ax=ax)
                ax.set_title("")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by temperature and model")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_boxplot_by_temperature_model.png", dpi=200)
            plt.close(fig)

            mean_df = (
                metric_df.groupby(["model_key", "temperature"], dropna=False)[metric]
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            if sns:
                sns.lineplot(
                    data=mean_df,
                    x="temperature",
                    y=metric,
                    hue="model_key",
                    marker="o",
                    ax=ax,
                )
            else:
                for model_key, model_group in mean_df.groupby("model_key"):
                    ax.plot(
                        model_group["temperature"],
                        model_group[metric],
                        marker="o",
                        label=str(model_key),
                    )
                ax.legend(title="model_key")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(f"Mean {metric}")
            ax.set_title(f"Mean {metric} by temperature and model")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_mean_by_temperature_model.png", dpi=200)
            plt.close(fig)
        elif has_temp:
            fig, ax = plt.subplots(figsize=(8, 5))
            mean_df = metric_df.groupby("temperature", dropna=False)[metric].mean().reset_index()
            ax.plot(mean_df["temperature"], mean_df[metric], marker="o")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(f"Mean {metric}")
            ax.set_title(f"Mean {metric} by temperature")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_mean_by_temperature.png", dpi=200)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(metric_df[metric], bins=20)
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            ax.set_title(f"{metric} distribution")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_distribution.png", dpi=200)
            plt.close(fig)

def plot_entropy_means(summary_df: pd.DataFrame, output_dir: Path, entropy_cols: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping entropy plots (matplotlib unavailable): {exc}")
        return

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plots_dir = output_dir / "plots" / "entropy"
    plots_dir.mkdir(parents=True, exist_ok=True)

    has_temp = "temperature" in summary_df.columns
    has_model = "model_key" in summary_df.columns

    for metric in entropy_cols:
        if metric not in summary_df.columns:
            continue

        metric_df = summary_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue

        if has_temp and has_model:
            mean_df = (
                metric_df.groupby(["model_key", "temperature"], dropna=False)[metric]
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            if sns:
                sns.lineplot(
                    data=mean_df,
                    x="temperature",
                    y=metric,
                    hue="model_key",
                    marker="o",
                    ax=ax,
                )
            else:
                for model_key, model_group in mean_df.groupby("model_key"):
                    ax.plot(
                        model_group["temperature"],
                        model_group[metric],
                        marker="o",
                        label=str(model_key),
                    )
                ax.legend(title="model_key")
            ax.set_xlabel("Temperature")
            ax.set_ylabel(metric)
            ax.set_title(f"Mean {metric} by temperature and model")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_mean_by_temperature_model.png", dpi=200)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(metric_df[metric], bins=20)
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            ax.set_title(f"{metric} distribution")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_distribution.png", dpi=200)
            plt.close(fig)

def compute_occupation_model_summary(summary_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if "occupation" not in summary_df.columns or "model_key" not in summary_df.columns:
        return pd.DataFrame()

    available_metrics = [m for m in metric_cols if m in summary_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    grouped = summary_df.groupby(["occupation", "model_key"], dropna=False)
    agg_df = grouped[available_metrics].agg(["mean", "std", "median", "count"]).reset_index()
    flat_columns = []
    for col in agg_df.columns:
        if col in {"occupation", "model_key"}:
            flat_columns.append(col)
        elif isinstance(col, tuple):
            metric_name = col[0]
            stat_name = col[1] if len(col) > 1 else ""
            flat_columns.append(f"{metric_name}_{stat_name}".rstrip("_"))
        else:
            flat_columns.append(str(col))

    agg_df.columns = flat_columns

    return agg_df


def normalize_gender_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("none").astype(str).str.strip().str.lower()
    replacements = {
        "unspecified": "none",
        "none": "none",
        "nan": "none",
        "": "none",
    }
    return normalized.replace(replacements)


def compute_gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "model_key" not in df.columns or "gender" not in df.columns:
        return pd.DataFrame()

    gender_df = df.copy()
    gender_df["_gender"] = normalize_gender_series(gender_df["gender"])

    counts_df = (
        gender_df.groupby(["model_key", "_gender"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    counts_df["proportion"] = counts_df["count"] / counts_df.groupby("model_key")["count"].transform("sum")
    counts_df = counts_df.rename(columns={"_gender": "gender"})
    return counts_df


def compute_occupation_metrics_summary(summary_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if "occupation" not in summary_df.columns:
        return pd.DataFrame()

    metric_candidates = ["self_bleu", "semantic_div", "avg_mean_entropy"]
    available_metrics = [m for m in metric_candidates if m in summary_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    metrics_df = (
        summary_df.groupby("occupation", dropna=False)[available_metrics]
        .mean()
        .reset_index()
    )

    if "gender" not in raw_df.columns:
        metrics_df["female_rate"] = np.nan
        return metrics_df

    gender_df = raw_df.copy()
    gender_df["_gender"] = normalize_gender_series(gender_df["gender"])
    gender_df = gender_df[gender_df["_gender"].isin(["male", "female"])]
    if gender_df.empty:
        metrics_df["female_rate"] = np.nan
        return metrics_df

    gender_counts = (
        gender_df.groupby(["occupation", "_gender"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    gender_counts["female_rate"] = (
        gender_counts.get("female", 0)
        / (gender_counts.get("female", 0) + gender_counts.get("male", 0)).replace(0, np.nan)
    )
    metrics_df = metrics_df.merge(
        gender_counts[["occupation", "female_rate"]],
        on="occupation",
        how="left",
    )
    return metrics_df


def compute_occupation_model_metrics_summary(summary_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if "occupation" not in summary_df.columns or "model_key" not in summary_df.columns:
        return pd.DataFrame()

    metric_candidates = ["self_bleu", "semantic_div", "avg_mean_entropy"]
    available_metrics = [m for m in metric_candidates if m in summary_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    metrics_df = (
        summary_df.groupby(["occupation", "model_key"], dropna=False)[available_metrics]
        .mean()
        .reset_index()
    )

    if "gender" not in raw_df.columns:
        metrics_df["female_rate"] = np.nan
        return metrics_df

    gender_df = raw_df.copy()
    gender_df["_gender"] = normalize_gender_series(gender_df["gender"])
    gender_df = gender_df[gender_df["_gender"].isin(["male", "female"])]
    if gender_df.empty:
        metrics_df["female_rate"] = np.nan
        return metrics_df

    gender_counts = (
        gender_df.groupby(["occupation", "model_key", "_gender"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    gender_counts["female_rate"] = (
        gender_counts.get("female", 0)
        / (gender_counts.get("female", 0) + gender_counts.get("male", 0)).replace(0, np.nan)
    )
    metrics_df = metrics_df.merge(
        gender_counts[["occupation", "model_key", "female_rate"]],
        on=["occupation", "model_key"],
        how="left",
    )
    return metrics_df


def compute_occupation_model_differences(
    occupation_model_summary_df: pd.DataFrame,
    metric_cols: list[str],
    base_label: str = "base",
    sft_label: str = "sft",
) -> pd.DataFrame:
    if occupation_model_summary_df.empty:
        return pd.DataFrame()

    base_df = occupation_model_summary_df[occupation_model_summary_df["model_key"] == base_label].copy()
    sft_df = occupation_model_summary_df[occupation_model_summary_df["model_key"] == sft_label].copy()
    if base_df.empty or sft_df.empty:
        return pd.DataFrame()

    merged = base_df.merge(sft_df, on="occupation", suffixes=("_base", "_sft"), how="inner")

    for metric in metric_cols:
        base_mean_col = f"{metric}_mean_base"
        sft_mean_col = f"{metric}_mean_sft"
        if base_mean_col not in merged.columns or sft_mean_col not in merged.columns:
            continue

        merged[f"delta_{metric}_sft_minus_base"] = merged[sft_mean_col] - merged[base_mean_col]

        base_std_col = f"{metric}_std_base"
        sft_std_col = f"{metric}_std_sft"
        base_n_col = f"{metric}_count_base"
        sft_n_col = f"{metric}_count_sft"
        if all(col in merged.columns for col in [base_std_col, sft_std_col, base_n_col, sft_n_col]):
            pooled = np.sqrt(
                (
                    (merged[base_n_col] - 1) * np.square(merged[base_std_col].fillna(0.0))
                    + (merged[sft_n_col] - 1) * np.square(merged[sft_std_col].fillna(0.0))
                )
                / (merged[base_n_col] + merged[sft_n_col] - 2).clip(lower=1)
            )
            merged[f"cohens_d_{metric}"] = np.where(
                pooled > 0,
                merged[f"delta_{metric}_sft_minus_base"] / pooled,
                np.nan,
            )

    return merged


def plot_occupation_metrics_heatmap(metrics_df: pd.DataFrame, output_dir: Path, metric_cols: list[str]) -> None:
    if metrics_df.empty:
        print("Skipping occupation metrics heatmap (metrics table is empty)")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping occupation difference plots (matplotlib unavailable): {exc}")
        return

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plots_dir = output_dir / "plots" / "occupation"
    plots_dir.mkdir(parents=True, exist_ok=True)

    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    if not available_cols:
        print("Skipping occupation metrics heatmap (no metric columns found)")
        return

    heatmap_df = metrics_df[["occupation", *available_cols]].dropna(subset=available_cols, how="all").copy()
    if heatmap_df.empty:
        return

    heatmap_df = heatmap_df.set_index("occupation")[available_cols]
    heatmap_df = heatmap_df.sort_values(by=available_cols[0])

    fig_height = max(8, 0.28 * len(heatmap_df))
    fig_width = max(8, 1.2 * len(available_cols))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if sns:
        sns.heatmap(
            heatmap_df,
            cmap="viridis",
            linewidths=0.2,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "Mean metric value"},
        )
    else:
        im = ax.imshow(heatmap_df.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(available_cols)))
        ax.set_xticklabels(available_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        fig.colorbar(im, ax=ax, label="Mean metric value")

    ax.set_title("Occupation-level mean metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Occupation")
    fig.tight_layout()
    fig.savefig(plots_dir / "occupation_metrics_heatmap.png", dpi=200)
    plt.close(fig)


def plot_occupation_model_metrics_heatmap(metrics_df: pd.DataFrame, output_dir: Path, metric_cols: list[str]) -> None:
    if metrics_df.empty:
        print("Skipping occupation model metrics heatmap (metrics table is empty)")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping occupation model metrics heatmap (matplotlib unavailable): {exc}")
        return

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plots_dir = output_dir / "plots" / "occupation"
    plots_dir.mkdir(parents=True, exist_ok=True)

    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    if not available_cols:
        print("Skipping occupation model metrics heatmap (no metric columns found)")
        return

    heatmap_parts = []
    for metric in available_cols:
        pivot_df = metrics_df.pivot_table(
            index="occupation",
            columns="model_key",
            values=metric,
            aggfunc="mean",
        )
        if pivot_df.empty:
            continue
        pivot_df.columns = [f"{col}_{metric}" for col in pivot_df.columns]
        heatmap_parts.append(pivot_df)

    if not heatmap_parts:
        return

    heatmap_df = pd.concat(heatmap_parts, axis=1)
    heatmap_df = heatmap_df.sort_values(by=heatmap_df.columns[0])

    fig_height = max(8, 0.28 * len(heatmap_df))
    fig_width = max(10, 0.8 * len(heatmap_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if sns:
        sns.heatmap(
            heatmap_df,
            cmap="viridis",
            linewidths=0.2,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "Mean metric value"},
        )
    else:
        im = ax.imshow(heatmap_df.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        fig.colorbar(im, ax=ax, label="Mean metric value")

    ax.set_title("Occupation-level metrics by model")
    ax.set_xlabel("Metric (model_key)")
    ax.set_ylabel("Occupation")
    fig.tight_layout()
    fig.savefig(plots_dir / "occupation_model_metrics_heatmap.png", dpi=200)
    plt.close(fig)


def compute_occupation_delta_metrics(
    occupation_model_metrics_df: pd.DataFrame,
    base_label: str = "base",
    sft_label: str = "sft",
) -> pd.DataFrame:
    if occupation_model_metrics_df.empty:
        return pd.DataFrame()

    metric_candidates = ["self_bleu", "semantic_div", "avg_mean_entropy", "female_rate"]
    available_metrics = [m for m in metric_candidates if m in occupation_model_metrics_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    delta_series = {}
    base_female_rate = None
    sft_female_rate = None
    for metric in available_metrics:
        pivot_df = occupation_model_metrics_df.pivot_table(
            index="occupation",
            columns="model_key",
            values=metric,
            aggfunc="mean",
        )
        if base_label not in pivot_df.columns or sft_label not in pivot_df.columns:
            continue
        delta_series[f"delta_{metric}_sft_minus_base"] = pivot_df[sft_label] - pivot_df[base_label]
        if metric == "female_rate":
            base_female_rate = pivot_df[base_label]
            sft_female_rate = pivot_df[sft_label]

    if base_female_rate is not None and sft_female_rate is not None:
        delta_series["female_rate_base"] = base_female_rate
        delta_series["female_rate_sft"] = sft_female_rate

    if not delta_series:
        return pd.DataFrame()

    delta_df = pd.DataFrame(delta_series)
    delta_df = delta_df.reset_index()
    return delta_df


def plot_occupation_delta_metrics_heatmap(delta_df: pd.DataFrame, output_dir: Path) -> None:
    if delta_df.empty:
        print("Skipping occupation delta heatmap (delta table is empty)")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping occupation delta heatmap (matplotlib unavailable): {exc}")
        return

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plots_dir = output_dir / "plots" / "occupation"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [col for col in delta_df.columns if col != "occupation"]
    if not metric_cols:
        return

    heatmap_df = delta_df.set_index("occupation")[metric_cols]
    sort_col = "delta_female_rate_sft_minus_base" if "delta_female_rate_sft_minus_base" in heatmap_df.columns else metric_cols[0]
    heatmap_df = heatmap_df.sort_values(by=sort_col, ascending=False)

    fig_height = max(8, 0.28 * len(heatmap_df))
    fig_width = max(10, 1.2 * len(metric_cols))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if sns:
        sns.heatmap(
            heatmap_df,
            cmap="coolwarm",
            center=0,
            linewidths=0.2,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "Delta (SFT - Base) / female rate"},
        )
    else:
        im = ax.imshow(heatmap_df.values, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        fig.colorbar(im, ax=ax, label="Delta (SFT - Base) / female rate")

    ax.set_title("Occupation-level metric deltas (SFT - Base)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Occupation")
    fig.tight_layout()
    fig.savefig(plots_dir / "occupation_delta_metrics_heatmap.png", dpi=200)
    plt.close(fig)

def load_test():
    filename = "sft_test.json"
    filepath = results_dir / filename
    df = pd.read_json(filepath)

    return {"sft_test": df}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyword",
        default="assumed",
        help="Keyword to filter files in results_dir when --input is not set",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"Optional path to a single results file (json or jsonl). Default is keyword-based loading in {results_dir}",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()    
    
    # Load data
    input_path = Path(args.input) if args.input else assumed_results
    data_frames = load_results_files(results_dir, file_name_keyword=args.keyword, input_path=input_path)
    if not data_frames:
        raise ValueError(f"No files loaded from {results_dir} with keyword '{args.keyword}'")

    metric_dfs = {}
    occupation_summary_dfs = {}
    occupation_metrics_dfs = {}
    occupation_model_metrics_dfs = {}
    occupation_delta_metrics_dfs = {}
    gender_summary_dfs = {}
    tqdm.pandas(desc="Applying metrics by group")

    ensure_nltk_punkt()

    # Initialize shared models once and reuse for all groups/dataframes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    metrics_aggregator = partial(
        apply_all_metrics,
        semantic_model=semantic_model,
    )

    # go through one df at a time
    for k, df in tqdm(data_frames.items(), total=len(data_frames), desc="Processing files"):
        print(f"### Processing file [{k}]...")
        df = expand_entropy_columns(df)

        group_cols = [col for col in ["model_key", "profile_id", "temperature", "prompt_case"] if col in df.columns]
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

        metrics_long = (
            df.groupby(group_cols, dropna=False)["response"]
            .progress_apply(metrics_aggregator)
            .reset_index()
        )

        # progress_apply + reset_index returns long format with metric names in `level_...`
        # and their values in `response`; pivot to wide format (self_bleu/semantic_div columns).
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

        print(f"Summary columns for [{k}]: {list(summary_df.columns)}")

        metric_k = "metrics_summary_" + k
        metric_dfs[metric_k] = summary_df

        gender_summary_df = compute_gender_distribution(df)
        if not gender_summary_df.empty:
            gender_summary_key = "gender_summary_" + k
            gender_summary_dfs[gender_summary_key] = gender_summary_df

        metric_cols = [
            "self_bleu",
            "semantic_div",
            *[f"avg_{col}" for col in entropy_cols],
        ]

        occupation_summary_df = compute_occupation_model_summary(summary_df, metric_cols=metric_cols)
        if not occupation_summary_df.empty:
            occupation_summary_key = "occupation_summary_" + k
            occupation_summary_dfs[occupation_summary_key] = occupation_summary_df

        occupation_metrics_df = compute_occupation_metrics_summary(summary_df, df)
        if not occupation_metrics_df.empty:
            occupation_metrics_key = "occupation_metrics_" + k
            occupation_metrics_dfs[occupation_metrics_key] = occupation_metrics_df

        occupation_model_metrics_df = compute_occupation_model_metrics_summary(summary_df, df)
        if not occupation_model_metrics_df.empty:
            occupation_model_metrics_key = "occupation_model_metrics_" + k
            occupation_model_metrics_dfs[occupation_model_metrics_key] = occupation_model_metrics_df

        occupation_delta_metrics_df = compute_occupation_delta_metrics(occupation_model_metrics_df)
        if not occupation_delta_metrics_df.empty:
            occupation_delta_metrics_key = "occupation_delta_metrics_" + k
            occupation_delta_metrics_dfs[occupation_delta_metrics_key] = occupation_delta_metrics_df

        if not args.no_plots:
            plot_metrics(summary_df, results_dir, ["self_bleu", "semantic_div"])
            if entropy_cols:
                plot_entropy_means(
                    summary_df,
                    results_dir,
                    [f"avg_{col}" for col in entropy_cols],
                )
            plot_occupation_delta_metrics_heatmap(
                occupation_delta_metrics_df,
                results_dir,
            )
    
    # save dict to json using util function
    save_dataframes(metric_dfs, results_dir)
    if occupation_summary_dfs:
        save_dataframes(occupation_summary_dfs, results_dir)
    if occupation_metrics_dfs:
        save_dataframes(occupation_metrics_dfs, results_dir)
    if occupation_model_metrics_dfs:
        save_dataframes(occupation_model_metrics_dfs, results_dir)
    if occupation_delta_metrics_dfs:
        save_dataframes(occupation_delta_metrics_dfs, results_dir)
    if gender_summary_dfs:
        save_dataframes(gender_summary_dfs, results_dir)