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

import json
import argparse
from nltk.tokenize import word_tokenize

# ------------ HELPER FUNCTIONS ------------
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
    
def slice_and_truncate(text, occupations: list, tokenizer, max_tokens=512, window_total=100):
    """
        Slice the input text into chunks that fit within the max token limit.
        Identify all occurrences of all occupations in the list, and create window_total-token windows around each.
    """
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    if len(input_ids) <= max_tokens:
        return text  # No truncation needed

    # Find locations of occupation occurrences in the text
    # Occupation tokens will always be found because the data are filtered already
    occupation_token_ids = [tokenizer(occ, add_special_tokens=False).input_ids for occ in occupations]
    occurrences = []
    for occ_ids in occupation_token_ids:
        occ_len = len(occ_ids)
        if occ_len == 0:
            continue
        for i in range(len(input_ids) - occ_len + 1):
            if input_ids[i:i + occ_len] == occ_ids:
                occurrences.append((i, occ_len))

    if not occurrences:
        truncated_text = tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True)
        return truncated_text

    occurrences.sort(key=lambda item: item[0])
    desired_context = [max(window_total - occ_len, 0) for _, occ_len in occurrences]

    def build_windows(context_sizes):
        windows = []
        for (pos, occ_len), ctx in zip(occurrences, context_sizes):
            left = ctx // 2
            right = ctx - left
            start = max(0, pos - left)
            end = min(len(input_ids), pos + occ_len + right)
            target_len = occ_len + ctx
            current_len = end - start
            if current_len < target_len:
                extra = target_len - current_len
                extend_right = min(extra, len(input_ids) - end)
                end += extend_right
                extra -= extend_right
                if extra > 0:
                    start = max(0, start - extra)
            windows.append((start, end))

        windows.sort(key=lambda item: item[0])
        merged = []
        for start, end in windows:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        merged_tokens = []
        for start, end in merged:
            merged_tokens.extend(input_ids[start:end])
        return merged_tokens

    merged_tokens = build_windows(desired_context)
    if len(merged_tokens) > max_tokens and sum(desired_context) > 0:
        total_desired_context = sum(desired_context)
        occ_len_sum = sum(occ_len for _, occ_len in occurrences)
        budget = max(max_tokens - occ_len_sum, 0)
        scale = min(1.0, budget / total_desired_context) if total_desired_context else 0.0
        context_sizes = [int(math.floor(ctx * scale)) for ctx in desired_context]
        remaining = max(budget - sum(context_sizes), 0)
        if remaining > 0:
            for idx, ctx in enumerate(context_sizes):
                if remaining == 0:
                    break
                gap = desired_context[idx] - ctx
                if gap <= 0:
                    continue
                add = min(gap, remaining)
                context_sizes[idx] += add
                remaining -= add
        merged_tokens = build_windows(context_sizes)

    if len(merged_tokens) > max_tokens:
        merged_tokens = merged_tokens[:max_tokens]

    truncated_text = tokenizer.decode(merged_tokens, skip_special_tokens=True)
    return truncated_text

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

def count_tokens(text, tokenizer):
    """Helper function to count the number of tokens in a given text string."""
    tokens = tokenizer(text, return_tensors="pt")
    return tokens.input_ids.shape[1]

if __name__ == "__main__":
    # Setup pipeline for zero-shot classification
    candidate_gender = ["male", "female", "none"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model and tokenizer from cache
    model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            cache_dir=models_dir,
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
    args = parser.parse_args()
    # convert testing arg to bool
    args.testing = args_to_bool(args.testing)

    labelled_data = load_labelled_dolci(testing=args.testing)
    output_path = dolci_classified_file
    i = 1
    truncated = 0
    progress_interval = 5000 if not args.testing else 10
    with open(output_path, "w", encoding="utf-8") as f:
        for row in labelled_data:
            occupations = pipe_separate_professions(row['professions'])
            if not occupations:
                truncated += 1
                continue            
            id = row['original_index']
            sample = row["messages"]
            if isinstance(sample, str):
                sample = json.loads(sample)
            content = ""
            # Get all the user (instruct) messages in one string
            for doc in sample:
                #TODO: Come up with a strategy to deal with roles since they are no longer in the filtered dataset
                if doc.get('role') == 'user':
                    total_tokens = count_tokens(content + doc['content'], tokenizer)
                    if total_tokens < 512:
                        content += doc['content'] + " "
                    else: 
                        truncated += 1
                        print(f"Clipping sample {id} (original_index) due to token limit. Token count with new content: {total_tokens}")
                        content = slice_and_truncate(content + doc['content'], occupations, tokenizer)
            for occ in occupations:
                hypothesis_template = f"The {occ} in the sample text is {{}}."
                output = pipe(content, candidate_gender, hypothesis_template=hypothesis_template, multi_label=False)
                classification = output['labels'][0]
                score = output['scores'][0]
                all_results = {"id": id, "occupation": occ, "sequence": content[:150], "classification": classification, "score": score}
                f.write(json.dumps(all_results, ensure_ascii=False) + '\n')
                i += 1
                if i % progress_interval == 0:
                    print(f"Processed {i} samples. Truncated {truncated} samples.")
                              
    print(f"Classified {i} samples. Truncated {truncated} samples.\nResults written to: {output_path}")