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

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
    # Setup pipeline for zero-shot classification
    candidate_gender = ["male", "female", "none"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model and tokenizer from cache
    model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            cache_dir=models_dir,
                                                            trust_remote_code=True,
                                                            token=HF_TOKEN).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              cache_dir=models_dir,
                                              token=HF_TOKEN)

    pipe = pipeline("zero-shot-classification", 
                    model=model,
                    tokenizer=tokenizer,
                    device=device)  
    
    # collect args and load samples
    parser = argparse.ArgumentParser(description="Run gender classification on MNLI test samples")
    parser.add_argument("--testing", default=0, help="Whether to run in testing mode with a smaller sample of the data")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per profession for zero-shot inference")
    args = parser.parse_args()
    # convert testing arg to bool
    args.testing = args_to_bool(args.testing)
    batch_size = max(int(args.batch_size), 1)

    labelled_data = load_labelled_dolci(testing=args.testing)
    print(f"Loaded {len(labelled_data)} labelled samples from Dolci SFT dataset with occupation labels.")
    output_path = dolci_classified_file
    i = 1
    truncated = 0
    progress_interval = 5000 if not args.testing else 10

    rows = []
    df_occtext = pd.DataFrame()
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

    # Now the new df has a row for each profession and a corresponding word-token window.
    # Pass batches to the pipeline.
    with open(output_path, "w", encoding="utf-8") as f:
        for prof, group in tqdm(df_occtext.groupby("prof"), desc="Running batched zero-shot classification by profession"):
            texts = group["text"].tolist()
            ids = group["id"].tolist()
            hypothesis_template = f"The {prof} in the sample text is {{}}."
            
            results = pipe(
                texts,
                candidate_gender,
                hypothesis_template=hypothesis_template,
                multi_label=False,
                batch_size=batch_size
            )
            
            for id_, text, result in zip(ids, texts, results):
                record = {
                    "id": id_,
                    "occupation": prof,
                    "classification": result["labels"][0],
                    "score": result["scores"][0]
                }
                f.write(json.dumps(record) + "\n")
    print(f"Classification complete. Results saved to {output_path}")