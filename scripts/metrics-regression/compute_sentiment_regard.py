"""
Compute per-response Sentiment and Regard scores for OLMo-3-7B "given" results.

Splits each `response` into sentences, runs two HuggingFace text-classification
pipelines over all sentences (batched across the whole dataset), and aggregates
the per-sentence scores back into per-response columns:

- avg_sentiment: mean of normalised [-1, 1] sentiment scores
  (siebert/sentiment-roberta-large-english)
- regard_<label> for each label returned by sasha/regardv3
  (negative/neutral/other/positive), plus regard_polarity = positive - negative
- n_sentences: number of valid sentences used for the averages
"""

from pathlib import Path
import sys
import json
import argparse
from datetime import datetime, timezone

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm
from transformers import pipeline

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

results_dir = root_dir / "data" / "olmo7b_results"
default_input = results_dir / "olmo7b_temp_results_given.jsonl"

SENTIMENT_MODEL = "siebert/sentiment-roberta-large-english"
REGARD_MODEL = "sasha/regardv3"


def ensure_nltk_punkt() -> None:
    try:
        word_tokenize("test")
        sent_tokenize("test sentence.")
    except LookupError:
        import nltk

        print("NLTK punkt not found. Downloading...")
        nltk.download("punkt")
        nltk.download("punkt_tab")


def read_results_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def extract_sentences(response) -> list[str]:
    """Split a response into sentences for "higher resolution" scoring.

    Falls back to splitting on newlines if the response comes back as a
    single long blob (e.g. the model didn't produce sentence-final
    punctuation), and drops very short fragments left by truncated generations.
    """
    if not isinstance(response, str) or not response.strip():
        return []

    sentences = sent_tokenize(response)

    if len(sentences) == 1 and len(sentences[0]) > 500:
        sentences = [s.strip() for s in response.split("\n") if s.strip()]

    return [s for s in sentences if len(s) > 20]


def normalise_sentiment(sentiment_score: dict) -> float:
    """Map a {label, score} dict to a signed score in [-1, 1]."""
    label = sentiment_score["label"]
    score = sentiment_score["score"]
    return -score if label.upper() == "NEGATIVE" else score


def build_sentence_index(responses: pd.Series) -> tuple[list[int], list[str]]:
    row_indices: list[int] = []
    sentence_texts: list[str] = []
    for row_idx, response in enumerate(responses):
        for sentence in extract_sentences(response):
            row_indices.append(row_idx)
            sentence_texts.append(sentence)
    return row_indices, sentence_texts


def run_pipeline_batched(pipe, texts: list[str], batch_size: int, desc: str) -> list:
    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        chunk = texts[start : start + batch_size]
        results.extend(pipe(chunk, truncation=True, max_length=512))
    return results


def aggregate_sentiment(row_indices: list[int], sentiment_results: list[dict], n_rows: int) -> pd.Series:
    scores = [normalise_sentiment(r) for r in sentiment_results]
    grouped = pd.Series(scores, index=row_indices, dtype=float).groupby(level=0).mean()
    return grouped.reindex(range(n_rows))


def aggregate_regard(row_indices: list[int], regard_results: list[list[dict]], n_rows: int) -> tuple[pd.DataFrame, list[str]]:
    per_sentence = pd.DataFrame(
        [{item["label"].lower(): item["score"] for item in sentence_scores} for sentence_scores in regard_results],
        index=row_indices,
    )

    labels = sorted(per_sentence.columns.tolist())

    grouped = per_sentence.groupby(level=0).mean().reindex(range(n_rows))
    grouped.columns = [f"regard_{c}" for c in grouped.columns]

    pos_col = next((c for c in grouped.columns if "positive" in c), None)
    neg_col = next((c for c in grouped.columns if "negative" in c), None)
    if pos_col and neg_col:
        grouped["regard_polarity"] = grouped[pos_col] - grouped[neg_col]

    return grouped, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input, help="Path to a results .jsonl/.json file")
    parser.add_argument("--output", type=Path, default=None, help="Output .jsonl path (default: <input_stem>_with_sentiment_regard.jsonl next to input file)")
    parser.add_argument("--metadata-dir", type=Path, default=None, help="Directory for metadata JSON (default: metrics_<input_stem>/ next to output file)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records (for smoke testing)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_idx = 0 if device == "cuda" else -1
    print(f"Using device: {device}")

    output_path = args.output or args.input.parent / f"{args.input.stem}_with_sentiment_regard.jsonl"

    print(f"Loading {args.input}")
    df = read_results_file(args.input)
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)
    n_rows = len(df)
    print(f"Loaded {n_rows} records")

    ensure_nltk_punkt()

    row_indices, sentence_texts = build_sentence_index(df["response"])
    print(f"Extracted {len(sentence_texts)} sentences from {n_rows} responses")

    pipeline_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

    print(f"Loading sentiment model: {SENTIMENT_MODEL}")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=device_idx,
        **pipeline_kwargs,
    )

    print(f"Loading regard model: {REGARD_MODEL}")
    regard_pipe = pipeline(
        "text-classification",
        model=REGARD_MODEL,
        top_k=None,
        device=device_idx,
        **pipeline_kwargs,
    )

    sentiment_results = run_pipeline_batched(sentiment_pipe, sentence_texts, args.batch_size, "Sentiment")
    regard_results = run_pipeline_batched(regard_pipe, sentence_texts, args.batch_size, "Regard")

    sentence_counts = pd.Series(row_indices, dtype="int64").value_counts()
    df["n_sentences"] = sentence_counts.reindex(range(n_rows)).fillna(0).astype(int).values

    df["avg_sentiment"] = aggregate_sentiment(row_indices, sentiment_results, n_rows).values

    regard_df, regard_labels = aggregate_regard(row_indices, regard_results, n_rows)
    print(f"Discovered regard labels: {regard_labels}")
    for col in regard_df.columns:
        df[col] = regard_df[col].values

    n_rows_zero_sentences = int((df["n_sentences"] == 0).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Wrote {n_rows} records to {output_path}")

    metadata_dir = args.metadata_dir or output_path.parent / f"metrics_{args.input.stem}"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_path": str(args.input),
        "output_path": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_records": n_rows,
        "n_sentences": len(sentence_texts),
        "n_rows_zero_sentences": n_rows_zero_sentences,
        "sentiment_model": SENTIMENT_MODEL,
        "regard_model": REGARD_MODEL,
        "regard_labels": regard_labels,
        "batch_size": args.batch_size,
        "device": device,
        "limit": args.limit,
    }
    metadata_path = metadata_dir / "sentiment_regard_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
