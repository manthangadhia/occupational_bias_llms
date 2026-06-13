print("Extracting gender assumption from base-model robustness narratives...")
import os
import argparse
from pathlib import Path
cwd = Path(__file__).parent
root_dir = cwd.parent                # .py < scripts < root > models, data
robustness_results_dir = root_dir / "data" / "robustness_results"

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


def build_hypothesis(prompt_style: str, occupation, label: str) -> str:
    """Build the entailment hypothesis for a narrative, tailored to the prompt style.

    Most styles carry an occupation and we probe the gender of the person holding that
    occupation. The baseline styles carry no occupation, so we just probe the gender of
    the narrative's subject directly: an anthropomorphic frog ("frog") or a person
    ("generic").
    """
    if prompt_style == "frog":
        return f"The frog in this response is {label}."
    if prompt_style == "generic":
        return f"The person in this response is {label}."
    return f"The person with occupation {occupation} in this response is {label}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract gender assumptions from base-model robustness narratives via MNLI entailment."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="base_robustness_results.jsonl",
        help="Filename within data/robustness_results/ to process (default: base_robustness_results.jsonl).",
    )
    args = parser.parse_args()
    robustness_results_file = robustness_results_dir / args.input
    if not robustness_results_file.exists():
        raise FileNotFoundError(f"Robustness results file not found: {robustness_results_file}")

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
    Example input format:
    {"model_key": "apertus-8b", "model_name": "swiss-ai/Apertus-8B-2509", "prompt_style": "assumed", "profile_id": 0, "temperature": 0.2, "occupation": "doctor", "attended_university": "no", "response_number": 1, "response": " a woman named Dr. Maria Gonzalez. She is a Mexican-American woman...", "entropy_analysis": {"mean_entropy": 0.135, "max_entropy": 1.328, "min_entropy": 5.39e-31, "std_entropy": 0.269, "mean_entropy_nucleus": 0.103, "max_entropy_nucleus": 1.266, "min_entropy_nucleus": -0.0, "std_entropy_nucleus": 0.243}}
    """

    # Load the base-model robustness narratives
    results_df = pd.read_json(robustness_results_file, lines=True)
    total = len(results_df)
    narratives = results_df["response"].tolist()
    # "occupation" is absent for occupation-free styles (e.g. frog, generic); default to None.
    if "occupation" in results_df.columns:
        occupations = results_df["occupation"].tolist()
    else:
        occupations = [None] * total
    prompt_styles = results_df["prompt_style"].tolist()
    gender = []
    scores = []
    for content, occupation, prompt_style in tqdm(
        zip(narratives, occupations, prompt_styles), total=total, desc="Processing narratives"
    ):
        premises = [content] * len(candidate_gender)
        hypotheses = [
            build_hypothesis(prompt_style, occupation, label)
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
    results_df.to_json(robustness_results_file, orient="records", lines=True)
    print(f"Results written to: {robustness_results_file}")
