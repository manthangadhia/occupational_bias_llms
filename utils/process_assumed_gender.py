#utils/process_assumed_gender.py
import pandas as pd
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
import argparse

# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
prompts_dir = data_dir / "gender_prompts"
results_dir = data_dir / "olmo7b_results"

def get_gender_assignment(assumed_results_df, batch_size):
    # Filter out rows with empty or null responses
    valid_mask = assumed_results_df['response'].notna() & (assumed_results_df['response'].str.strip() != '')
    valid_df = assumed_results_df[valid_mask].reset_index(drop=False)
    
    print(f"Total rows: {len(assumed_results_df)}, Valid responses: {len(valid_df)}, Empty/null responses: {len(assumed_results_df) - len(valid_df)}")
    
    # Convert the valid DataFrame to a Hugging Face Dataset
    ds = Dataset.from_pandas(valid_df[['response']])
    
    # Initialize the pipeline on the GPU (device=0) and use batch_size=32
    print(f"Cuda is available: {torch.cuda.is_available()}")
    classifier = pipeline(
        "zero-shot-classification", 
        model="MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size 
    )

    # Define gender candidate labels and hypothesis statement for model classification
    candidate_labels = ["male", "female", "unspecified"]
    hypothesis_template = "The writer assumed a gender for the character. This gender is {}."

    # Run inference
    results = []
    for out in tqdm(classifier(KeyDataset(ds, "response"), 
                        candidate_labels=candidate_labels, 
                        hypothesis_template=hypothesis_template),
                    total=len(ds),
                    desc="Classifying gender"):
        # 'out' is a dict; we just want the top label
        all_res_zip = zip(out['labels'], out['scores'])
        print(f"Labels and scores: {dict(all_res_zip)}")  # Debug print to see the full output structure
        results.append(out['labels'][0])

    # Add results back to the original DataFrame
    # Initialize with a default value for all rows
    assumed_results_df['gender'] = 'unspecified'
    # Map the results back to valid rows using the preserved original indices
    original_indices = valid_df['index'].values
    for i, gender_label in enumerate(results):
        assumed_results_df.loc[original_indices[i], 'gender'] = gender_label

    return assumed_results_df

def load_assumed_files(results_dir: Path) -> dict:
    """Load all assumed output files from the results directory and return as Dataframe in dict"""
    # Get all assumed files from the results dir
    all_assumed_files = list(results_dir.glob("*assumed*.json"))
    print("Found the following files in results_dir:")
    for file in all_assumed_files:
        print(file.name)
    # Make dict with each file as a df and then convert to Dataset for HF pipeline
    all_assumed_dfs = {file.name: pd.read_json(file) for file in all_assumed_files}

    return all_assumed_dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process assumed gender assumption.")
    parser.add_argument("--batch_size",
                        default=24,
                        type=int,
                        help="Batch size for rows to process in the HF pipeline. Adjust based on GPU memory.")
    args = parser.parse_args()
    # Get a dict with each assumed output file stored as a DataFrame
    all_assumed_dfs = load_assumed_files(results_dir)
    

    for name, df in all_assumed_dfs.items():
        print(f"\nProcessing file: {name} with shape {df.shape}")
        all_assumed_dfs[name] = get_gender_assignment(df, batch_size=args.batch_size)
        
        # Save the updated DataFrame with gender assignments
        output_path = results_dir / str(name)
        all_assumed_dfs[name].to_json(output_path, orient='records', force_ascii=True, indent=4)
        print(f"✓ Saved results to {output_path.name}")
