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

# Special sets for KW search
# (need extra processing/checking throughout the pipeline)
whole_word_required = set({     # if these words are found, check that they are whole words!
    "dj", ""
    "cop"})
ambiguous_professions = set({   # if these words are found, do POS tagging and ensure noun
    "coach",
    "cook",
    "guard",
    "judge",
    "nurse",
    "pilot",
    "doctor",
    "broker",
    "tutor",
    "captain",
    "minister",
    "soldier",
    "steward",
    "advocate",
    "clerk",
    "engineer",
    "planner",
    "porter", })

import datasets
from datasets import load_dataset
from tqdm import tqdm
import re

# For kw search
import ahocorasick
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])

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
    return data.get("professions", [])

def load_dolci_data(data_path: Path) -> datasets.Dataset:
    """Load the Dolci-SFT dataset from the given parquet file path."""
    return load_dataset("parquet", data_files=str(data_path))["train"]

def pipe_join_professions(professions_lists) -> str:
    """Join a list of professions into a single string for easier searching."""
    # This will return an empty string if an empty list is passed --- this is the desired behaviour
    return "|".join(professions_lists)

def pipe_split_professions(professions_str: str) -> list:
    """Split a string of professions back into a list."""
    if professions_str == "":
        return []
    return professions_str.split("|")

def is_whole_word(text, start, end):
    """Check that match boundaries are not adjacent to word characters."""
    before_ok = (start == 0) or (not text[start - 1].isalpha())
    after_ok = (end == len(text) - 1) or (not text[end + 1].isalpha())
    return before_ok and after_ok

def find_professions_in_text(A: ahocorasick.Automaton, 
                             text: str, original_text: str,
                             whole_word_required=whole_word_required,
                             ambiguous_professions=ambiguous_professions) -> list:
    """Use the Aho-Corasick automaton to find all professions mentioned in the given text."""
    found_professions = set()
    ambiguous_hits = set()  # to track which ambiguous professions were hit for later checking
    for end_index, (prof_index, prof) in A.iter(text):
        start_index = end_index - len(prof) + 1
        if prof in whole_word_required and not is_whole_word(text, start_index, end_index):
            continue  # skip substring matches for this keyword
        if prof in ambiguous_professions:
            ambiguous_hits.add(prof)
        else:
            found_professions.add(prof)
        
    # if this sample has any ambiguous professions, then tokenise text once and do pos tagging
    if ambiguous_hits:
        doc = nlp(original_text)
        noun_tokens = {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}
        for prof in ambiguous_hits:
            if prof in noun_tokens:
                found_professions.add(prof)

    return sorted(found_professions)

def search_dolci_for_professions():    
    # Load all professions and dolci data
    professions = get_professions(professions_file)
    print(f"Loaded {len(professions)} professions.")
    dolci_data = load_dolci_data(dolci_dataset)

    total_samples = len(dolci_data)
    print(f"Loaded Dolci-SFT dataset with {total_samples} samples.")

    # Create Aho-Corasick automaton for efficient keyword searching
    A = ahocorasick.Automaton()
    for i, prof in enumerate(professions):
        A.add_word(prof.lower(), (i, prof))
    A.make_automaton()

    # Across all dolci samples, label each sample with the mentioned and filter
    instruct_professions = []
    response_professions = []
    all_professions = []        # this is to keep track of all professions in that entry, across instruct and response

    for row in tqdm(dolci_data):
        instruct_prof_temp = []
        response_prof_temp = []
        for turn in row["messages"]:
            role = turn["role"]
            content = turn["content"]
            if content is None:
                continue
            original_text = content  # keep the original text for POS tagging
            text = content.lower()  # lowercase for matching
            if role == "user":
                instruct_prof_temp += find_professions_in_text(A, text=text, original_text=original_text)
            elif role == "assistant":
                response_prof_temp += find_professions_in_text(A, text=text, original_text=original_text)

        # combine instruct and response into a set for tracking all professions
        all_prof_set = set(instruct_prof_temp + response_prof_temp)
        # convert all lists/sets to pipe-joined strings for storage
        instruct_prof_str = pipe_join_professions(instruct_prof_temp)
        response_prof_str = pipe_join_professions(response_prof_temp)
        all_prof_str = pipe_join_professions(all_prof_set)

        instruct_professions.append(instruct_prof_str)
        response_professions.append(response_prof_str)
        all_professions.append(all_prof_str)

    assert len(instruct_professions) == total_samples, f"Mismatch in number of samples and instruct professions: {len(instruct_professions)} != {total_samples}"
    assert len(response_professions) == total_samples, f"Mismatch in number of samples and response professions: {len(response_professions)} != {total_samples}"
    assert len(all_professions) == total_samples, f"Mismatch in number of samples and all professions: {len(all_professions)} != {total_samples}"

    # Convert ds to df and add professions to dataframe and save
    dolci_df = pd.DataFrame()
    dolci_df = dolci_data.to_pandas()
    dolci_df["original_index"] = dolci_df.index
    dolci_df["instruct_professions"] = instruct_professions
    dolci_df["response_professions"] = response_professions
    dolci_df["all_professions"] = all_professions

    # Filter dolci df to samples where there are at least some professions mentioned
    dolci_df = dolci_df[
        (dolci_df["all_professions"].apply(lambda x: len(x) > 0))
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
    
    # process the profession strings into lists and count the frequency of each profession in instruct vs response
    from collections import Counter
    instruct_prof_counter = Counter()
    response_prof_counter = Counter()
    for prof_str in instruct_professions:
        prof_list = pipe_split_professions(prof_str)
        instruct_prof_counter.update(prof_list)
    for prof_str in response_professions:
        prof_list = pipe_split_professions(prof_str)
        response_prof_counter.update(prof_list)
    # are all 303 professions represented? which ones are not represented at all?
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