from pathlib import Path
import pandas as pd
import re
import sys

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

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass

@dataclass
class FightinWordsResult:
    vocab: list[str]
    log_odds: np.ndarray       # z-scored log-odds ratio for each word; positive = more associated with group_a
    z_scores: np.ndarray
    counts_a: np.ndarray
    counts_b: np.ndarray

def fightin_words(
    texts_a: list[str],
    texts_b: list[str],
    prior_texts: list[str] | None = None,
    alpha: float = 0.01,
    max_features: int = 10_000,
    z_threshold: float = 1.96,
) -> FightinWordsResult:
    """
    Monroe et al. (2008) Fightin' Words with informative Dirichlet prior.
    
    Computes z-scored log-odds ratios for words distinguishing two corpora.
    Positive z-scores indicate association with texts_a; negative with texts_b.

    Args:
        texts_a:      Corpus A (e.g. male-labelled responses)
        texts_b:      Corpus B (e.g. female-labelled responses)
        prior_texts:  Background corpus for the Dirichlet prior.
                      If None, uses texts_a + texts_b (uninformative).
        alpha:        Prior scaling factor (smaller = less smoothing)
        max_features: Vocabulary size cap for CountVectorizer
        z_threshold:  Significance threshold; used only to filter results

    Returns:
        FightinWordsResult with z-scores and counts for each vocab item.
        Filter on result.z_scores > z_threshold for significant words.
    """
    prior_corpus = prior_texts if prior_texts is not None else texts_a + texts_b

    # Fit vocabulary on the full prior corpus
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(prior_corpus)
    vocab = vectorizer.get_feature_names_out().tolist()

    # Count word frequencies in each corpus
    counts_a = np.asarray(vectorizer.transform(texts_a).sum(axis=0)).flatten()
    counts_b = np.asarray(vectorizer.transform(texts_b).sum(axis=0)).flatten()
    prior_counts = np.asarray(vectorizer.transform(prior_corpus).sum(axis=0)).flatten()

    # Informative Dirichlet prior: alpha * (prior word freq / total prior freq)
    total_prior = prior_counts.sum()
    alpha_w = alpha * (prior_counts / total_prior)  # shape: (vocab_size,)

    # Smoothed totals
    n_a = counts_a.sum()
    n_b = counts_b.sum()
    alpha_0 = alpha_w.sum()

    # Log-odds ratio with prior smoothing (Monroe et al. eq. 17)
    log_odds = (
        np.log(counts_a + alpha_w) - np.log(n_a + alpha_0 - counts_a - alpha_w)
      - np.log(counts_b + alpha_w) + np.log(n_b + alpha_0 - counts_b - alpha_w)
    )

    # Variance estimate (Monroe et al. eq. 22)
    variance = (
        1.0 / (counts_a + alpha_w)
      + 1.0 / (counts_b + alpha_w)
    )

    z_scores = log_odds / np.sqrt(variance)

    return FightinWordsResult(
        vocab=vocab,
        log_odds=log_odds,
        z_scores=z_scores,
        counts_a=counts_a,
        counts_b=counts_b,
    )

def get_significant_words(
    result: FightinWordsResult,
    z_threshold: float = 1.96,
    top_n: int | None = 30,
) -> dict[str, list[tuple[str, float]]]:
    """
    Extract significant words from a FightinWordsResult.

    Returns a dict with keys 'group_a' and 'group_b', each containing
    a list of (word, z_score) tuples sorted by descending |z_score|.
    """
    sig_mask = np.abs(result.z_scores) > z_threshold
    sig_indices = np.where(sig_mask)[0]

    group_a = [(result.vocab[i], result.z_scores[i]) for i in sig_indices if result.z_scores[i] > 0]
    group_b = [(result.vocab[i], result.z_scores[i]) for i in sig_indices if result.z_scores[i] < 0]

    group_a.sort(key=lambda x: -x[1])
    group_b.sort(key=lambda x: x[1])

    if top_n:
        group_a = group_a[:top_n]
        group_b = group_b[:top_n]

    return {"group_a": group_a, "group_b": group_b}


def read_results_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def clean_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.replace("\r\n", "\n")
    match = re.search(r"\nassistant\s*\n", normalized, flags=re.IGNORECASE)
    if match:
        normalized = normalized[match.end():]
    return " ".join(normalized.split())


def extract_texts(series: pd.Series) -> list[str]:
    return [text for text in series if isinstance(text, str) and text.strip()]

# ---------------------------------------------------------------------------
# Usage sketch — replace with your actual data loading
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = read_results_file(assumed_results)

    required_cols = {"response", "gender"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df["response_clean"] = df["response"].apply(clean_response_text)

    gender_series = df["gender"].fillna("").astype(str).str.strip().str.lower()
    texts_male = extract_texts(df.loc[gender_series == "male", "response_clean"])
    texts_female = extract_texts(df.loc[gender_series == "female", "response_clean"])
    all_texts = extract_texts(df["response_clean"])

    if not texts_male or not texts_female:
        raise ValueError(
            "Need at least one male and one female response to run fightin_words."
        )

    # --- CORE COMPUTATION ---
    result = fightin_words(
        texts_a=texts_male,
        texts_b=texts_female,
        prior_texts=all_texts,   # or None to use only male+female as prior
    )
    significant = get_significant_words(result, z_threshold=1.96, top_n=30)

    print("Words more associated with MALE-labelled responses:")
    for word, z in significant["group_a"]:
        print(f"  {word:20s}  z={z:.2f}")

    print("\nWords more associated with FEMALE-labelled responses:")
    for word, z in significant["group_b"]:
        print(f"  {word:20s}  z={z:.2f}")

    output_dir = results_dir / "fighting_words_outputs"
    output_key = f"{assumed_results.stem}_with_clean"
    save_dataframes({output_key: df}, output_dir)
    print(f"Saved updated dataframe to {output_dir / (output_key + '.json')}")