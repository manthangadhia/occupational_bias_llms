print("Testing gender mask pipeline with sample document...")
import math
import os
from pathlib import Path
cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < scripts < occ_bias < root > models         # structure on euler
project_dir = root_dir / "occ_bias"
dolci_dir = project_dir / "data" / "dolci_sft"
professions_file = dolci_dir / "filtered_professions.json"      # json with 299 professions combined from debiswe and 100 years of stereotypes
dolci_dataset = dolci_dir / "dolci_sft.parquet"                 # full, original dolci-sft dataset
dolci_professions_file = dolci_dir / "dolci_sft_with_professions.parquet"
dolci_classified_file = dolci_dir / "dolci_mnli_gender_classification.jsonl"
staged_models = os.getenv("HF_HOME") or os.getenv("MODEL_ROOT")
models_dir = Path(staged_models) if staged_models else (root_dir / "models" / "bert")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Model cache directory: {models_dir}")

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datasets as ds
import pandas as pd
from functools import lru_cache
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=6, progress_bar=True)
MAX_WORD_TOKENS = 480

import json
import argparse
from tqdm import tqdm
import gc

# ------------ HELPER FUNCTIONS ------------
@lru_cache(maxsize=1)
def _get_detokenizer():
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    return TreebankWordDetokenizer()

def args_to_bool(arg):
    """Convert a string argument to a boolean."""
    arg = str(arg).lower()
    if isinstance(arg, bool):
        return arg
    if arg in ("yes", "true", "t", "y", "1"):
        return True
    elif arg in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def process_text_length(text: str, occupation: str, max_words: int = MAX_WORD_TOKENS):
    """
        Slice the input text into chunks that fit within the word-token limit.
        For each occupation:
        1. Identify all occurrences of the occupation in the text.
        2. If multiple occurrences, take the "middle" occurrence, and if no occurrence, return an empty string.
        3. Create a window of `max_words` size around this occurrence (including the occupation tokens).
        4. Return the window as a string.
    """
    from nltk.tokenize import word_tokenize
    detokenizer = _get_detokenizer()

    text_tokens = word_tokenize(text)
    num_tokens = len(text_tokens)
    if num_tokens <= max_words:
        return text  # No truncation needed

    occupation_tokens = word_tokenize(occupation)
    occurrences = []

    occ_len = len(occupation_tokens)
    if occ_len == 0:
        return ""

    for i in range(len(text_tokens) - occ_len + 1):
        if text_tokens[i:i + occ_len] == occupation_tokens:
            occurrences.append((i, occ_len))

    if not occurrences:
        return ""

    occurrences.sort(key=lambda item: item[0])
    if len(occurrences) > 1:
        # If multiple occurrences, take the "middle" one
        middle_index = len(occurrences) // 2
        start_pos, occ_len = occurrences[middle_index]
    else:
        start_pos, occ_len = occurrences[0]  # Only one occurrence

    occ_center = start_pos + (occ_len // 2)
    desired_start = occ_center - (max_words // 2)
    max_start = max(len(text_tokens) - max_words, 0)
    window_start = max(0, min(desired_start, max_start))
    window_end = min(window_start + max_words, len(text_tokens))
    window_tokens = text_tokens[window_start:window_end]

    return detokenizer.detokenize(window_tokens)


def get_entailment_index(model: AutoModelForSequenceClassification) -> int:
    """Resolve the entailment logit index from the model config if available."""
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}

    # Prefer label2id if it includes an entailment key
    for key, value in label2id.items():
        if "entail" in str(key).lower():
            return int(value)

    # Fall back to id2label if needed
    for key, value in id2label.items():
        if "entail" in str(value).lower():
            return int(key)

    # Final fallback by num_labels
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels == 3:
        return 2
    if num_labels == 2:
        return 1

    raise ValueError(f"Unable to resolve entailment index for num_labels={num_labels}")


def get_entailment_scores(logits: torch.Tensor, entailment_idx: int) -> torch.Tensor:
    """
    Given raw NLI logits of shape (batch_size, num_labels),
    return only the entailment column, shape (batch_size,).
    """
    return logits[:, entailment_idx]


def pipe_separate_professions(professions_str):
    """Convert a pipe-separated string of professions back into a list."""
    if professions_str:
        return professions_str.split("|")
    else:
        return []

TESTING_SAMPLE_SIZE = 100

def load_labelled_dolci(testing: bool = False):
    infile = dolci_professions_file
    dataset = ds.load_dataset("parquet", data_files=str(infile))["train"]
    if testing:
        dataset = dataset.select([i for i in list(range(TESTING_SAMPLE_SIZE))])
    return dataset

if __name__ == "__main__":
    # collect args and load samples
    parser = argparse.ArgumentParser(description="Run gender classification on MNLI test samples")
    parser.add_argument("--testing", default=0, help="Whether to run in testing mode with a smaller sample of the data")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per profession for zero-shot inference")
    parser.add_argument("--do-filtering", default=1, help="Whether to run the text windowing/filtering step")
    parser.add_argument("--do-classify", default=1, help="Whether to run zero-shot classification")
    args = parser.parse_args()
    # convert testing arg to bool
    args.testing = args_to_bool(args.testing)
    args.do_filtering = args_to_bool(args.do_filtering)
    args.do_classify = args_to_bool(args.do_classify)
    batch_size = max(int(args.batch_size), 1)

    if not (args.do_filtering or args.do_classify):
        raise ValueError("At least one of --do-filtering or --do-classify must be true.")

    output_path = dolci_classified_file
    i = 1
    truncated = 0
    progress_interval = 5000 if not args.testing else 10

    df_occtext = pd.DataFrame()
    if args.do_filtering:
        labelled_data = load_labelled_dolci(testing=args.testing)
        print(f"Loaded {len(labelled_data)} labelled samples from Dolci SFT dataset with occupation labels.")

        rows = []
        for row in tqdm(labelled_data, total=len(labelled_data), desc="Processing occupations and content"):
            original_id = int(row["original_index"])
            content = ""
            # get all turns in the row, and all instruct content from each turn
            turns = json.loads(row["messages"])
            content = " ".join(
                t["content"] for t in turns
                if t.get("role") == "user" and t.get("content")
            )
            professions = pipe_separate_professions(row["professions"])
            for prof in professions:
                prof = prof.lower().strip()
                # processed_text = process_text_length(content, prof, tokenizer, occ_input_id)
                # if processed_text == "":
                #     continue
                rows.append({
                    "id": original_id,
                    "prof": prof,
                    "text": content
                })
        df_occtext = pd.DataFrame(rows)
        # parallelapply the text processing function to the dataframe
        df_occtext["text"] = df_occtext.parallel_apply(lambda x: process_text_length(x["text"], x["prof"]), axis=1)
        # filter out rows where text is empty after processing
        df_occtext = df_occtext[df_occtext["text"] != ""]
        print(f"After processing text length, {len(df_occtext)} rows remain for classification. {len(rows) - len(df_occtext)} rows were removed due to empty text after processing.")
        del rows
        gc.collect()

        # save filteredd and processed dataframe to disk for inspection
        df_occtext.to_parquet(dolci_professions_file, engine="pyarrow")
        print(f"Saved processed dataframe with professions and text to {dolci_professions_file}")
    else:
        if not dolci_professions_file.exists():
            raise FileNotFoundError(
                f"Processed file not found: {dolci_professions_file}. Run with --do-filtering=1 first."
            )
        df_occtext = pd.read_parquet(dolci_professions_file)
        required_columns = {"id", "prof", "text"}
        missing_columns = required_columns - set(df_occtext.columns)
        if missing_columns:
            raise ValueError(f"Processed file missing required columns: {sorted(missing_columns)}")
        print(f"Loaded {len(df_occtext)} processed rows from {dolci_professions_file}")

    if args.do_classify:
        if df_occtext.empty:
            raise ValueError("No rows available for classification. Run with --do-filtering=1 first.")

        # Setup pipeline for zero-shot classification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Define model and tokenizer from cache
        model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=models_dir,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=models_dir,
            token=HF_TOKEN
        )
        model.eval()
        entailment_idx = get_entailment_index(model)

        candidate_gender = ["male", "female", "none"]
        num_labels = len(candidate_gender)

        # Sort by profession so same-profession rows stay together (good for memory locality)
        df_occtext = df_occtext.sort_values("prof").reset_index(drop=True)

        # Build flat lists
        all_texts = df_occtext["text"].tolist()
        all_profs = df_occtext["prof"].tolist()
        all_ids = df_occtext["id"].tolist()

        total_rows = len(all_texts)

        with open(output_path, "w", encoding="utf-8") as f:
            for batch_start in tqdm(
                range(0, total_rows, batch_size),
                desc="Running manual batched classification"
            ):
                batch_end = min(batch_start + batch_size, total_rows)

                batch_texts = all_texts[batch_start:batch_end]
                batch_profs = all_profs[batch_start:batch_end]
                batch_ids = all_ids[batch_start:batch_end]

                # For each (text, profession) pair, create one premise-hypothesis pair
                # per candidate label — so N rows * 3 candidates = 3N NLI pairs total.
                premises = []
                hypotheses = []

                for text, prof in zip(batch_texts, batch_profs):
                    for label in candidate_gender:
                        premises.append(text)
                        hypotheses.append(f"The {prof} in the sample text is {label}.")

                # Tokenise all pairs together
                encoded = tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation="only_first",
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    logits = model(**encoded).logits

                # Extract entailment scores and reshape to (N, num_labels)
                entailment_scores = get_entailment_scores(logits, entailment_idx)
                entailment_scores = entailment_scores.view(-1, num_labels)

                # Softmax across the candidate labels for each row
                probs = torch.softmax(entailment_scores, dim=-1)
                best_label_indices = probs.argmax(dim=-1).tolist()
                best_scores = probs.max(dim=-1).values.tolist()

                for id_, prof, label_idx, score in zip(
                    batch_ids,
                    batch_profs,
                    best_label_indices,
                    best_scores
                ):
                    record = {
                        "id": id_,
                        "occupation": prof,
                        "classification": candidate_gender[label_idx],
                        "score": round(float(score), 6)
                    }
                    f.write(json.dumps(record) + "\n")
        print(f"Classification complete. Results saved to {output_path}")