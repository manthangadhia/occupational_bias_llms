"""
Script to count the frequency of all professions and save their counts in a json.

Update 2026-05-26: I added this function to dolci_occupation_search.py to compute the counts on the fly. Separate script is no longer needed.
"""
import pandas as pd
from pathlib import Path
import json
# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent.parent                  # .py < scripts < occ_bias < root > data         # structure on euler
project_dir = root_dir / "occ_bias"
dolci_dir = project_dir / "data" / "dolci_sft"
professions_file = dolci_dir / "select_professions.json"        # json with 303 professions combined from debiswe and 100 years of stereotypes
dolci_dataset = dolci_dir / "dolci_sft.parquet"                 # full, original dolci-sft dataset
dolci_professions_file = dolci_dir / "dolci_sft_with_professions.parquet"

import pandas as pd

# ------- HELPER FUNCTIONS --------
def get_professions(professions_file: Path) -> list:
    """Load the list of professions from the given JSON file path."""
    with professions_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("professions", [])

def pipe_split_professions(professions_str: str) -> list:
    """Split a string of professions back into a list."""
    if professions_str == "":
        return []
    return professions_str.split("|")

def compute_profession_stats():
    """
    •	check which occupations are present (all? some? which are not present?)
    •	check the most frequently represented occupations. 
    o	then run a gender signal nli check on top_k most present occupations. 
    """
    # Load dolci df with professions
    dolci_df = pd.read_parquet(dolci_professions_file, engine="pyarrow")
    # get the list of professions
    professions = dolci_df["professions"].tolist()

    
    # process the profession strings into lists and count the frequency of each profession
    from collections import Counter
    prof_counter = Counter()
    for profession_str in professions:
        prof_list = pipe_split_professions(profession_str)
        prof_counter.update(prof_list)

    counts_output_path = dolci_dir / "dolci_profession_counts.json"
    sorted_counts = [{"profession": p, "count": c} for p, c in prof_counter.most_common()]
    with counts_output_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_counts, f, indent=2, ensure_ascii=False)
    print(f"Saved profession counts to {counts_output_path}")

if __name__ == "__main__":
    compute_profession_stats()