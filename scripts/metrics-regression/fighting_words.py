"""
fighting_words.py

Monroe et al. (2008) Fightin' Words analysis comparing male vs female responses.

Within each group (e.g., per model_key × occupation), computes z-scored log-odds ratios
that identify words statistically over-represented in female vs male responses.

Output convention:
  - Positive z-score  = word more associated with FEMALE responses
  - Negative z-score  = word more associated with MALE responses

Output JSON schema (one record per group):
  {
    "group": {"model_key": "base", "occupation": "nurse"},
    "n_female": 120,
    "n_male": 15,
    "top_female_words": [["she", 5.2], ["her", 4.8]],   # z > 0, sorted descending
    "top_male_words":   [["he", 4.5], ["his", 3.9]],    # stored as positive strengths
    "all_words": [                                        # top 200 words by |z|
      {"word": "she", "z_score": 5.2, "count_female": 340, "count_male": 12},
      ...
    ],
    "vocab_size": 5000
  }

Usage:
  # Given/assumed (group by model × occupation):
  python fighting_words.py --input data/olmo7b_results/foo_with_sentiment_regard.jsonl

  # Robustness (group by model × prompt_style, pooling occupations):
  python fighting_words.py --input data/robustness_results/bar.jsonl --group-by model_key,prompt_style
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Core algorithm (Monroe et al. 2008) — unchanged from original
# ---------------------------------------------------------------------------

@dataclass
class FightinWordsResult:
    vocab: list[str]
    log_odds: np.ndarray   # signed log-odds; positive = more in group_a
    z_scores: np.ndarray   # z-scored log-odds
    counts_a: np.ndarray   # word counts in corpus A
    counts_b: np.ndarray   # word counts in corpus B


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
        texts_a:      Corpus A
        texts_b:      Corpus B
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

    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(prior_corpus)
    vocab = vectorizer.get_feature_names_out().tolist()

    counts_a      = np.asarray(vectorizer.transform(texts_a).sum(axis=0)).flatten()
    counts_b      = np.asarray(vectorizer.transform(texts_b).sum(axis=0)).flatten()
    prior_counts  = np.asarray(vectorizer.transform(prior_corpus).sum(axis=0)).flatten()

    total_prior = prior_counts.sum()
    alpha_w = alpha * (prior_counts / total_prior)

    n_a     = counts_a.sum()
    n_b     = counts_b.sum()
    alpha_0 = alpha_w.sum()

    log_odds = (
        np.log(counts_a + alpha_w) - np.log(n_a + alpha_0 - counts_a - alpha_w)
      - np.log(counts_b + alpha_w) + np.log(n_b + alpha_0 - counts_b - alpha_w)
    )
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

    Returns a dict with keys 'group_a' (z > threshold) and 'group_b' (z < -threshold),
    each a list of (word, z_score) tuples sorted by descending |z_score|.
    """
    sig_mask = np.abs(result.z_scores) > z_threshold
    sig_indices = np.where(sig_mask)[0]

    group_a = [(result.vocab[i], result.z_scores[i]) for i in sig_indices if result.z_scores[i] > 0]
    group_b = [(result.vocab[i], result.z_scores[i]) for i in sig_indices if result.z_scores[i] < 0]

    group_a.sort(key=lambda x: -x[1])
    group_b.sort(key=lambda x:  x[1])

    if top_n:
        group_a = group_a[:top_n]
        group_b = group_b[:top_n]

    return {"group_a": group_a, "group_b": group_b}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.replace("\r\n", "\n")
    match = re.search(r"\nassistant\s*\n", normalized, flags=re.IGNORECASE)
    if match:
        normalized = normalized[match.end():]
    return " ".join(normalized.split())


def extract_texts(series: pd.Series) -> list[str]:
    return [t for t in series if isinstance(t, str) and t.strip()]


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Monroe et al. Fightin' Words, comparing male vs female responses per group. "
            "Positive z-score = female-associated; negative z-score = male-associated."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input JSONL with 'response' and 'gender' columns.",
    )
    parser.add_argument(
        "--group-by", type=str, default="model_key,occupation",
        help=(
            "Comma-separated columns for grouping (default: model_key,occupation). "
            "For robustness files use: model_key,prompt_style"
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Top N words per gender to save (default 20).",
    )
    parser.add_argument(
        "--min-group-size", type=int, default=10,
        help="Skip groups where either gender has fewer than this many responses (default 10).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Defaults to {input_dir}/{stem}_fighting_words.json",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    output_path = args.output or (args.input.parent / f"{args.input.stem}_fighting_words.json")
    group_cols  = [c.strip() for c in args.group_by.split(",")]

    print(f"Loading {args.input} ...")
    df = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(df)} rows.")

    missing_gc = [c for c in group_cols if c not in df.columns]
    if missing_gc:
        raise ValueError(f"Group columns not in file: {missing_gc}")
    for col in ("response", "gender"):
        if col not in df.columns:
            raise ValueError(f"Required column missing: '{col}'")

    df["response_clean"] = df["response"].apply(clean_response_text)

    records = []
    skipped = 0

    for group_key, grp in tqdm(df.groupby(group_cols), desc="Groups"):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_dict = dict(zip(group_cols, group_key))

        g_col = grp["gender"].fillna("").astype(str).str.strip().str.lower()
        # texts_a = female, texts_b = male → positive z = female-associated
        texts_female = extract_texts(grp.loc[g_col == "female", "response_clean"])
        texts_male   = extract_texts(grp.loc[g_col == "male",   "response_clean"])

        if len(texts_female) < args.min_group_size or len(texts_male) < args.min_group_size:
            print(
                f"  Skipping {group_dict}: "
                f"n_female={len(texts_female)}, n_male={len(texts_male)} "
                f"(min={args.min_group_size})"
            )
            skipped += 1
            continue

        all_texts = texts_female + texts_male
        result = fightin_words(
            texts_a=texts_female,
            texts_b=texts_male,
            prior_texts=all_texts,
        )
        significant = get_significant_words(result, z_threshold=1.96, top_n=args.top_n)

        # group_a = female (z > 0); group_b = male (z < 0, store as positive strength)
        top_female_words = [[w, round(float(z),  4)] for w, z in significant["group_a"]]
        top_male_words   = [[w, round(float(-z), 4)] for w, z in significant["group_b"]]

        # top 200 words by |z| with raw signed z-score
        top200_idx = np.argsort(np.abs(result.z_scores))[::-1][:200]
        all_words = [
            {
                "word":         result.vocab[i],
                "z_score":      round(float(result.z_scores[i]), 4),  # +female, -male
                "count_female": int(result.counts_a[i]),
                "count_male":   int(result.counts_b[i]),
            }
            for i in top200_idx
        ]

        records.append({
            "group":            group_dict,
            "n_female":         len(texts_female),
            "n_male":           len(texts_male),
            "top_female_words": top_female_words,
            "top_male_words":   top_male_words,
            "all_words":        all_words,
            "vocab_size":       len(result.vocab),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(
        f"Written {len(records)} group records to {output_path} "
        f"({skipped} groups skipped due to min-group-size)."
    )


if __name__ == "__main__":
    main()
