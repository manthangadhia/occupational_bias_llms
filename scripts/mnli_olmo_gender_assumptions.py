print("Extracting gender assumption from Olmo NLG Narratives...")
import os
from pathlib import Path
cwd = Path(__file__).parent
root_dir = cwd.parent                # .py < scripts < occ_bias < root > models         # structure on euler
# project_dir = root_dir / "occ_bias"
# dolci_dir = root_dir / "data" / "dolci_sft"
# professions_file = dolci_dir / "select_professions.json"        # json with 299 professions combined from debiswe and 100 years of stereotypes
olmo_results_file = root_dir / "data" / "olmo7b_results" / "olmo7b_temp_results_assumed.jsonl"

# staged_models = os.getenv("HF_HOME") or os.getenv("MODEL_ROOT")

# models_dir = Path(staged_models) if staged_models else (root_dir / "models" / "bert")
models_dir = root_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Model cache directory: {models_dir}")

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

def get_entailment_index(model: AutoModelForSequenceClassification) -> int:
    """Resolve the entailment logit index from the model config if available."""
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}

    for key, value in label2id.items():
        if "entail" in str(key).lower():
            return int(value)

    for key, value in id2label.items():
        if "entail" in str(value).lower():
            return int(key)

    num_labels = getattr(model.config, "num_labels", None)
    if num_labels == 3:
        return 2
    if num_labels == 2:
        return 1

    raise ValueError(f"Unable to resolve entailment index for num_labels={num_labels}")

if __name__ == "__main__":
    candidate_gender = ["male", "female", "none"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model and tokenizer from cache
    model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    model_kwargs = {
        "cache_dir": models_dir,
        "token": HF_TOKEN
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              cache_dir=models_dir,
                                              token=HF_TOKEN)
    model.eval()
    entailment_idx = get_entailment_index(model)
    
    """
    Example output format:
    {"model_key": "sft", "model_name": "allenai/Olmo-3-7B-Instruct-SFT", "prompt_case": "assumed", "profile_id": 28, "temperature": 0.7, "occupation": "consultant", "attended_university": "yes", "response_number": 3, "response": " You are allowed to make assumptions about the person's personality, based on the provided characteristics. The occupation title should be mentioned in the response.\n\nassistant\nGrowing up in a bustling coastal town, I was always fascinated by the constant ebb and flow of ideas...", "entropy_analysis": {"mean_entropy": 0.9206281426341836, "max_entropy": 3.046875, "min_entropy": 1.3096723705530167e-10, "std_entropy": 0.7879512841699567}}

    """
    
    # Load the Olmo narratives dataset
    results_df = pd.read_json(olmo_results_file, lines=True)
    total = len(results_df)
    narratives = results_df["response"].tolist()
    occupations = results_df["occupation"].tolist()
    gender = []
    scores = []
    for content, occupation in tqdm(zip(narratives, occupations), total=total, desc="Processing narratives"):
        premises = [content] * len(candidate_gender)
        hypotheses = [
            f"The person with occupation {occupation} in this response is {label}."
            for label in candidate_gender
        ]

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

        entailment_scores = logits[:, entailment_idx]
        probs = torch.softmax(entailment_scores, dim=0)
        best_idx = int(probs.argmax().item())
        best_score = float(probs.max().item())

        gender.append(candidate_gender[best_idx])
        scores.append(best_score)

    results_df["gender"] = gender
    results_df["gender_score"] = scores
    results_df.to_json(olmo_results_file, orient="records", lines=True)                             
    print(f"Results written to: {olmo_results_file}")