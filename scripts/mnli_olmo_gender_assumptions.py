print("Extracting gender assumption from Olmo NLG Narratives...")
import os
from pathlib import Path
cwd = Path(__file__).parent
root_dir = cwd.parent.parent                # .py < scripts < occ_bias < root > models         # structure on euler
project_dir = root_dir / "occ_bias"
dolci_dir = project_dir / "data" / "dolci_sft"
professions_file = dolci_dir / "select_professions.json"        # json with 299 professions combined from debiswe and 100 years of stereotypes
olmo_results_file = project_dir / "data" / "olmo" / "olmo7_narratives.jsonl"

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
    
    # Load the Olmo narratives dataset
    # TODO
    narratives = []
    
    for narrative in narratives:
        occ = narrative['occupation']
        content = narrative['content']
        id = narrative['id']
        hypothesis_template = f"The {occ} in the sample text is {{}}."
        output = pipe(content, candidate_gender, hypothesis_template=hypothesis_template, multi_label=False)
        classification = output['labels'][0]
        score = output['scores'][0]
        all_results = {"id": id, "occupation": occ, "sequence": content[:150], "classification": classification, "score": score}
        f.write(json.dumps(all_results, ensure_ascii=False) + '\n')
        i += 1
        if i % progress_interval == 0:
            print(f"Processed {i} samples. Skipped {skipped} samples.")
                              
    print(f"Results written to: {output_path}")