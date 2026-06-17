"""
compute_diversity_metrics.py

Reads a JSONL file (typically *_with_sentiment_regard.jsonl), computes per-response-group
self-BLEU and semantic diversity, and broadcasts group-level values back to each row,
updating the JSONL in-place by default.

Self-BLEU: average BLEU of each response against its group-mates
  - Lower = responses are more repetitive / less diverse
Semantic diversity: average pairwise cosine distance of sentence-transformer embeddings
  - Higher = responses are more semantically varied

Groups are defined by the generation context columns (model_key, profile_id, temperature,
prompt_case/prompt_style) — i.e., all rows sharing the same context, differing only in
response_number. Groups of size 1 get NaN for both metrics.

Usage:
  # In-place enrichment (overwrites input):
  python compute_diversity_metrics.py --input data/.../foo_with_sentiment_regard.jsonl

  # Custom output path:
  python compute_diversity_metrics.py --input foo.jsonl --output foo_enriched.jsonl

  # Smoke test on first 50 rows:
  python compute_diversity_metrics.py --input foo.jsonl --limit 50

  # Explicit group-by (for robustness files with prompt_style):
  python compute_diversity_metrics.py --input robustness.jsonl --group-by model_key,profile_id,temperature,prompt_style
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Metric functions (logic from compute_metrics.py; copied here to avoid
# module-level path-resolution side-effects at import time)
# ---------------------------------------------------------------------------

def ensure_nltk_punkt() -> None:
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("test")
    except LookupError:
        import nltk
        print("NLTK punkt not found. Downloading...")
        try:
            nltk.download("punkt_tab")
        except Exception:
            nltk.download("punkt")


def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> float:
    """
    Average BLEU of each text against the other texts in the group.
    Lower = responses are more repetitive (less diverse).
    Returns NaN for groups of size < 2.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize

    if len(texts) < 2:
        return float("nan")

    tokenized = [word_tokenize(t.lower()) for t in texts]
    smoothing = SmoothingFunction().method1
    scores_per_n: dict[int, list[float]] = {n: [] for n in ns}

    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        for n in ns:
            weights = tuple(1 / n for _ in range(n))
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
            scores_per_n[n].append(score)

    mean_scores = [float(np.mean(scores_per_n[n])) for n in ns]
    # Geometric mean across n-gram orders
    return float(np.exp(np.mean(np.log(np.clip(mean_scores, 1e-10, None)))))


def semantic_diversity(model, texts: list[str]) -> float:
    """
    Average pairwise cosine distance of sentence-transformer embeddings.
    Higher = responses are more semantically varied.
    Returns NaN for groups of size < 2.
    """
    from scipy.spatial.distance import pdist

    if len(texts) < 2:
        return float("nan")

    embeddings = model.encode(texts, show_progress_bar=False)
    pairwise_distances = pdist(embeddings, metric="cosine")
    return float(np.mean(pairwise_distances))


# ---------------------------------------------------------------------------
# Group-key auto-detection
# ---------------------------------------------------------------------------

_GROUP_CANDIDATES = ["model_key", "profile_id", "temperature", "prompt_case", "prompt_style"]


def detect_group_cols(df: pd.DataFrame) -> list[str]:
    """Return generation-context columns present in the dataframe."""
    return [c for c in _GROUP_CANDIDATES if c in df.columns]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute self-BLEU and semantic diversity per response group "
            "and broadcast the group-level value to each row in the JSONL."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input JSONL file (any results file with a 'response' column).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path. Defaults to overwriting --input in place.",
    )
    parser.add_argument(
        "--group-by", type=str, default=None,
        help=(
            "Comma-separated column names defining response groups. "
            "Auto-detected if not set (model_key, profile_id, temperature, prompt_case/prompt_style)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Sentence-transformer encoding batch size (default 64).",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Directory for metadata JSON. Defaults to <output_dir>/metrics_<input_stem>/",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows for smoke testing.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    output_path  = args.output or args.input   # default: overwrite in place
    metadata_dir = args.metadata_dir or (output_path.parent / f"metrics_{args.input.stem}")

    # Load
    print(f"Loading {args.input} ...")
    df = pd.read_json(args.input, lines=True)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"Loaded {len(df)} rows.")

    # Determine group columns
    if args.group_by:
        group_cols = [c.strip() for c in args.group_by.split(",")]
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"--group-by columns not in file: {missing}")
    else:
        group_cols = detect_group_cols(df)
        if not group_cols:
            raise ValueError(
                "Could not auto-detect group columns. "
                "Expected at least one of: " + ", ".join(_GROUP_CANDIDATES)
            )
        print(f"Auto-detected group columns: {group_cols}")

    # NLTK punkt (needed inside self_bleu → word_tokenize)
    ensure_nltk_punkt()

    # Sentence-transformer model
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"Loading sentence-transformer (all-MiniLM-L6-v2) on {device} ...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Compute per-group metrics
    groups = list(df.groupby(group_cols))
    print(f"Computing metrics for {len(groups)} groups ...")
    results: dict[tuple, dict[str, float]] = {}
    n_singletons = 0

    for name, grp in tqdm(groups, desc="Groups"):
        if not isinstance(name, tuple):
            name = (name,)
        texts = [t for t in grp["response"].tolist() if isinstance(t, str) and t.strip()]
        if len(texts) < 2:
            n_singletons += 1
            results[name] = {"self_bleu": float("nan"), "semantic_div": float("nan")}
        else:
            results[name] = {
                "self_bleu":    self_bleu(texts),
                "semantic_div": semantic_diversity(st_model, texts),
            }

    # Broadcast group values to each row
    keys = df[group_cols].apply(tuple, axis=1)
    df["self_bleu"]    = keys.map(lambda k: results.get(k, {}).get("self_bleu",    float("nan")))
    df["semantic_div"] = keys.map(lambda k: results.get(k, {}).get("semantic_div", float("nan")))

    # Write enriched JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Written {len(df)} rows to {output_path}")

    # Write metadata
    metadata_dir.mkdir(parents=True, exist_ok=True)
    nan_bleu = int(df["self_bleu"].isna().sum())
    nan_sem  = int(df["semantic_div"].isna().sum())
    meta = {
        "input_path":            str(args.input.resolve()),
        "output_path":           str(output_path.resolve()),
        "created_at":            datetime.now(timezone.utc).isoformat(),
        "n_records":             len(df),
        "n_groups":              len(groups),
        "n_singleton_groups":    n_singletons,
        "group_by_cols":         group_cols,
        "self_bleu_nan_rate":    round(nan_bleu / len(df), 4) if len(df) else None,
        "semantic_div_nan_rate": round(nan_sem  / len(df), 4) if len(df) else None,
        "st_model":              "all-MiniLM-L6-v2",
        "limit":                 args.limit,
    }
    meta_path = metadata_dir / "diversity_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"Metadata written to {meta_path}\n"
        f"Groups: {len(groups)}, singletons: {n_singletons}, "
        f"self_bleu NaN: {nan_bleu}/{len(df)}, semantic_div NaN: {nan_sem}/{len(df)}"
    )


if __name__ == "__main__":
    main()
