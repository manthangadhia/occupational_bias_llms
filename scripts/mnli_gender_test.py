print("Testing gender mask pipeline with sample document...")
import os
from pathlib import Path
cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < scripts < occ_bias < root > models         # structure on euler
data_dir = root_dir / "occ_bias" / "data"
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

def load_samples_streaming(filename):
    infile = data_dir / filename
    dataset = ds.load_dataset("json", data_files=str(infile))["train"]
    return dataset

def label_match_in_doc(label, doc):
    # check if the label matched a whole word in the doc, if not, skip the sample
    # split doc into words and check for exact match with label
    words = doc.split()
    return label in words

def write_output(results, output_name):
    """
    pipeline results structure:
    {
    'sequence': 'Teacher:You are given a sentence in Spanish. Your job is to translate the Spanish ...', 
    'labels': ['female', 'male', 'ambiguous', 'none'], 
    'scores': [0.5180891752243042, 0.2921257019042969, 0.14652131497859955, 0.043263792991638184]
    }

    To this I have added the sample id and label, and clipped the text sequence to the first 150 chars.
    The final output structure I want is:
    {
    'id': {
        'occupation': 'teacher',
        'sequence': 'Teacher:You are given a sentence in Spanish. Your job is to translate the Spanish ...',
        'label': "",
        'score': :4f
        }
    """
    pass

if __name__ == "__main__":
    # Setup pipeline for zero-shot classification
    candidate_gender = ["male", "female", "none", "ambiguous"]
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
    parser.add_argument("--testing", type=bool, default=False, help="Whether to run in testing mode with a smaller sample of the data")
    args = parser.parse_args()
    
    sample_file = "mnli_test_samples_subset.jsonl" if args.testing else "mnli_test_samples.jsonl"
    print(f"Running test on subset: {args.testing}\nLoading samples from: {sample_file}\n")
    all_samples = load_samples_streaming(filename=sample_file)

    output_name = "mnli_gender_classification_basic_hits.jsonl"
    output_path = data_dir / output_name
    i = 1
    skipped = 0
    progress_interval = 5000 if not args.testing else 10
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            doc = sample['content']
            occupations = sample['labels']
            id = sample['id']

            for occ in occupations:
                # check if the label matched a whole word in the doc, if not, skip the sample
                if not label_match_in_doc(occ, doc):
                    skipped += 1
                    continue
                hypothesis_template = f"The {occ} in the sample text is {{}}."
                output = pipe(doc, candidate_gender, hypothesis_template=hypothesis_template, multi_label=False)
                classification = output['labels'][0]
                score = output['scores'][0]
                all_results = {"id": id, "occupation": occ, "sequence": doc[:150], "classification": classification, "score": score}
                f.write(json.dumps(all_results, ensure_ascii=False) + '\n')
                i += 1
                if i % progress_interval == 0:
                    print(f"Processed {i} samples. Skipped {skipped} samples.")
                              
    print(f"Classified {i} samples. Skipped {skipped} samples.\nResults written to: {output_path}")