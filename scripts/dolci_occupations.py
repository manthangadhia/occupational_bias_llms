import pandas as pd
from pathlib import Path
import json
# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent.parent                # .py < scripts < occ_bias < root > data         # structure on euler
project_dir = root_dir / "occ_bias"
dolci_dir = project_dir / "data" / "dolci_sft"
professions_file = dolci_dir / "debiaswe_professions.json"
dolci_dataset = dolci_dir / "dolci_sft.parquet"
dolci_professions_file = dolci_dir / "dolci_sft_with_professions.parquet"

import datasets
from datasets import load_dataset
from tqdm import tqdm

import argparse
import gc
# Args helper function
def arg_to_bool(arg: int) -> bool:
    """Convert an integer argument to a boolean."""
    if arg not in (0, 1):
        raise argparse.ArgumentTypeError("Argument must be 0 or 1")
    return bool(arg)

def get_professions(professions_file: Path) -> list:
    """Load the list of professions from the given JSON file path."""
    with professions_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [item[0].replace("_", " ") for item in data]

def load_dolci_data(data_path: Path) -> datasets.Dataset:
    """Load the Dolci-SFT dataset from the given parquet file path."""
    return load_dataset("parquet", data_files=str(data_path))["train"]

def search_dolci_for_professions() -> pd.DataFrame:
    # Load all professions and dolci data
    professions = get_professions(professions_file)
    print(f"Loaded {len(professions)} professions.")
    dolci_data = load_dolci_data(dolci_dataset)

    total_samples = len(dolci_data)

    # Across all dolci samples, label each sample with the mentioned and filter
    instruct_professions = []
    response_professions = []
    for row in tqdm(dolci_data):
        temp_instruct_profs = []
        temp_response_profs = []
        for turn in row["messages"]:
            content = turn["content"]
            if content is None:
                temp_instruct_profs.extend([])
                temp_response_profs.extend([])
                continue
            content = content.lower()  # lowercase for matching
            if turn["role"] == "user":
                temp_instruct_profs.extend([prof for prof in professions if prof in content])
            elif turn["role"] == "assistant":
                temp_response_profs.extend([prof for prof in professions if prof in content])
        instruct_professions.append(temp_instruct_profs)
        response_professions.append(temp_response_profs)

    assert len(instruct_professions) == total_samples, f"Mismatch in number of samples and instruct professions: {len(instruct_professions)} != {total_samples}"
    assert len(response_professions) == total_samples, f"Mismatch in number of samples and response professions: {len(response_professions)} != {total_samples}"

    # Convert ds to df and add professions to dataframe and save
    dolci_df = pd.DataFrame()
    dolci_df = dolci_data.to_pandas()
    dolci_df["instruct_professions"] = instruct_professions
    dolci_df["response_professions"] = response_professions

    # Filter dolci df to samples where either instruct/response occ columns are non empty || drop rows with two empty lists
    dolci_df = dolci_df[
        (dolci_df["instruct_professions"].apply(lambda x: len(x) > 0)) |
        (dolci_df["response_professions"].apply(lambda x: len(x) > 0))
    ]    
    print(f"Filtered to {len(dolci_df)} samples with at least one profession mentioned in either instruction or response. {len(dolci_df)/total_samples:.2%} of total samples retained.")

    # Save the updated dataframe with professions
    output_path = dolci_professions_file
    dolci_df.to_parquet(output_path, index=False)
    print(f"Saved updated Dolci-SFT dataframe with professions to {output_path}")

def compute_profession_stats():
    """
    •	check which occupations are present (all? some? which are not present?)
    •	check the most frequently represented occupations. 
    o	then run a gender signal nli check on top_k most present occupations. 
    """
    # Load dolci df with professions
    dolci_df = pd.read_parquet(dolci_professions_file, engine="fastparquet")
    # get the list of instruct and response professions
    instruct_professions = dolci_df["instruct_professions"].tolist()
    response_professions = dolci_df["response_professions"].tolist()
    # flatten these lists and count frequency.
    from collections import Counter
    instruct_prof_counter = Counter([prof for sublist in instruct_professions for prof in sublist])
    response_prof_counter = Counter([prof for sublist in response_professions for prof in sublist])
        # are all 320 professions represented? which ones are not represented at all?
    professions = get_professions(professions_file)
    instruct_prof_set = set(instruct_prof_counter.keys())
    response_prof_set = set(response_prof_counter.keys())
    all_prof_set = instruct_prof_set.union(response_prof_set)
        # what are the top_k most frequently mentioned professions in instruct vs response?
    top_k = 25
    top_instruct_profs = instruct_prof_counter.most_common(top_k)
    bottom_instruct_profs = instruct_prof_counter.most_common()[:-top_k-1:-1]
    top_response_profs = response_prof_counter.most_common(top_k)
    bottom_response_profs = response_prof_counter.most_common()[:-top_k-1:-1]

    # save these stats to a json file for analysis
    stats_output = {
        "total_unique_professions_in_instructions": len(instruct_prof_set),
        "total_unique_professions_in_responses": len(response_prof_set),
        "total_unique_professions_in_both": len(all_prof_set),
        "professions_not_mentioned_at_all_instructions": list(set(professions) - instruct_prof_set),
        "professions_not_mentioned_at_all_responses": list(set(professions) - response_prof_set),
        "top_k_professions_in_instructions": top_instruct_profs,
        "bottom_k_professions_in_instructions": bottom_instruct_profs,
        "top_k_professions_in_responses": top_response_profs,
        "bottom_k_professions_in_responses": bottom_response_profs,
    }
    stats_output_path = dolci_dir / "dolci_profession_stats.json"
    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=4, ensure_ascii=False)
    print(f"Saved profession stats to {stats_output_path}")
    for key, value in stats_output.items():
        print(f"{key}: {value}")

def main(args):
    if arg_to_bool(args.search):
        print("Searching Dolci for professions...")
        # Call the function to search Dolci for professions
        search_dolci_for_professions()
        gc.collect()
    if arg_to_bool(args.get_stats):
        print("Computing profession stats...")
        # Call the function to compute stats on the distribution of professions in Dolci
        compute_profession_stats()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default=0, type=int, help="Whether to search Dolci for professions.")
    parser.add_argument("--get-stats", default=0, type=int, help="Whether to compute stats on the distribution of professions in Dolci.")
    args = parser.parse_args()

    main(args)