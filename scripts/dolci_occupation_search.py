"""
The Aho-Corasick search works quite fast, taking ~5min to search the whole dataset. 
POS tagging is the computational bottleneck, but it is helpful to do it at this stage and filter out false positives.
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

# Special sets for KW search
# (need extra processing/checking throughout the pipeline)
whole_word_required = set({     # if these words are found, check that they are whole words!
    "dj",
    "cop",
    })
ambiguous_professions = set({   # if these words are found, do POS tagging and ensure noun
    "author",
    "coach",
    "cook",
    "guard",
    "judge",
    "nurse",
    "pilot",
    "broker",
    "tutor",
    "minister",
    "advocate",
    })

import datasets
from datasets import load_dataset
from tqdm import tqdm
import re

# For kw search
import ahocorasick
import spacy
pos_nlp = spacy.load("en_core_web_trf", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
spacy.prefer_gpu()

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

def remove_prof_from_pipe(professions_str: str, prof_to_remove: str) -> str:
    """Remove a specific profession from a pipe-separated string of professions."""
    professions = pipe_split_professions(professions_str)
    if prof_to_remove in professions:
        professions.remove(prof_to_remove)
    return pipe_join_professions(professions)

def is_whole_word(text, start, end):
    """Check that match boundaries are not adjacent to word characters."""
    before_ok = (start == 0) or (not text[start - 1].isalpha())
    after_ok = (end == len(text) - 1) or (not text[end + 1].isalpha())
    return before_ok and after_ok

def find_professions_in_text(A: ahocorasick.Automaton, 
                             text: str, 
                             row_idx: int,
                             ambiguous_labels: set,
                             whole_word_required=whole_word_required,
                             ambiguous_professions=ambiguous_professions) -> list:
    """Use the Aho-Corasick automaton to find all professions mentioned in the given text."""
    # all incoming text needs to be made lowercase
    text = text.lower()
    found_professions = set()
    ambiguous_hits = set()  # to track which ambiguous professions were hit for later checking
    for end_index, (prof_index, prof) in A.iter(text):
        start_index = end_index - len(prof) + 1
        if prof in whole_word_required and not is_whole_word(text, start_index, end_index):
            continue  # skip substring matches for this keyword
        if prof in ambiguous_professions:
            ambiguous_hits.add(prof)
        found_professions.add(prof)
    # if this sample has any ambiguous professions, edit mutable dict object
    if ambiguous_hits:
        ambiguous_labels.update(ambiguous_hits)   # add the ambiguous professions found in this sample to the set for that sample   
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
    all_professions = []        # this is to keep track of all professions in that entry, across instruct and response
    track_ambiguous_rows = {}   # to keep track of which rows had ambiguous profession hits for later checking

    for row_idx, row in enumerate(tqdm(dolci_data)):
        temp_found_professions = []
        all_content = ""                # this is refreshed per row, and stored at the end of the loop if we find some ambiguous labels in the text
        ambiguous_labels = set()        # this is a mutable object and keeps getting built and then refreshed per row
        for turn in row["messages"]:
            # role = turn["role"]               # I don't need this
            content = turn["content"]
            if content is None:
                continue
            all_content += content + " "
        if not all_content:
            # add an empty string to all_prof list to have the same len as dolci
            all_professions.append("")
            continue # because there was no content in this row

        # after collecting all the text, we can run one keyword search (lowercasing is done during processing)
        temp_found_professions += find_professions_in_text(A, text=all_content, row_idx=row_idx, ambiguous_labels=ambiguous_labels)

        # if we had some ambiguous label hits in this row
        if ambiguous_labels:
            temp_dict = {"labels": list(ambiguous_labels), "text": all_content}
            track_ambiguous_rows[row_idx] = temp_dict

        # convert temp list to set (avoid duplicates) and then to pipe-joined string for storage
        temp_found_set = set(temp_found_professions)
        all_prof_str = pipe_join_professions(temp_found_set)
        all_professions.append(all_prof_str)
    
    # Now manage all the rows where we had ambiguous hits! Check if the label is present as a noun, and remove the label from that row if not
    print(f"Found {len(track_ambiguous_rows.keys())} ({(len(track_ambiguous_rows.keys()) / total_samples)*100:2%}%) rows that need POS tagging")
    with open(dolci_dir / "ambiguous_rows_info.json", "w", encoding="utf-8") as f:
        json.dump(track_ambiguous_rows, f, indent=4, ensure_ascii=False)
    print(f"Saved info on ambiguous rows to {dolci_dir / 'ambiguous_rows_info.json'} for later POS tagging and checking.")
    
    # Now my dict is in the form: {row_id: {"labels": ambiguous_hits:list, "text": all_content}}
    # collect all rows of text and row ids for the ambiguous hits 
    all_ambiguous_texts = [(row["text"], row_id) for row_id, row in track_ambiguous_rows.items()]
    # check POS for each label, 
    for doc, row_id in pos_nlp.pipe(all_ambiguous_texts, batch_size=16, as_tuples=True):
        noun_tokens = {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}
        labels = track_ambiguous_rows[row_id]["labels"]     # this is a list
        int_row_id = int(row_id)  # int for indexing into the professions lists
        for l in labels:
        # and remove the label from the pipe if its fails the check 
            if l not in noun_tokens: # if the ambiguous profession is not used as a noun in the text, then we remove it from the pipe string for that row
                all_professions[int_row_id] = remove_prof_from_pipe(all_professions[int_row_id], l)

    assert len(all_professions) == total_samples, f"Mismatch in number of samples and all professions: {len(all_professions)} != {total_samples}"

    # Convert ds to df and add professions to dataframe and save
    dolci_df = pd.DataFrame()
    dolci_df = dolci_data.to_pandas()
    dolci_df["original_index"] = dolci_df.index
    dolci_df["professions"] = all_professions

    # Filter dolci df to samples where there are at least some professions mentioned
    dolci_df = dolci_df[
        (dolci_df["all_professions"].apply(lambda x: len(x) > 0))
    ]    
    print(f"Filtered to {len(dolci_df)} samples with at least one profession mentioned in either instruction or response. {len(dolci_df)/total_samples:.2%} of total samples retained.")

    # Save the updated dataframe with professions
    output_path = dolci_professions_file
    dolci_df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved updated Dolci-SFT dataframe with professions to {output_path}")

def compute_profession_stats():
    """
    •	check which occupations are present (all? some? which are not present?)
    •	check the most frequently represented occupations. 
    o	then run a gender signal nli check on top_k most present occupations. 
    """
    # Load dolci df with professions
    dolci_df = pd.read_parquet(dolci_professions_file, engine="pyarrow")
    dolci_df["messages"] = dolci_df["messages"].apply(json.loads)   # convert messages back from json
    # get the list of instruct and response professions
    instruct_professions = dolci_df["instruct_professions"].tolist()
    response_professions = dolci_df["response_professions"].tolist()

    # read ambiguous rows json
    with open(dolci_dir / "ambiguous_rows_info.json", "r", encoding="utf-8") as f:
        ambiguous_rows_data = json.load(f)
    
    # process the profession strings into lists and count the frequency of each profession in instruct vs response
    from collections import Counter
    instruct_prof_counter = Counter()
    response_prof_counter = Counter()
    ambiguous_counter = 0   # to track how many rows in my final filtered collection had ambiguous profession hits
    for idx, instruct_str, response_str in enumerate(zip(instruct_professions, response_professions)):
        instruct_prof_list = pipe_split_professions(instruct_str)
        instruct_prof_counter.update(instruct_prof_list)
        response_prof_list = pipe_split_professions(response_str)
        response_prof_counter.update(response_prof_list)
        # check if this row had an ambiguous profession hit and if so, update the ambiguous counter
        if idx in ambiguous_rows_data:   # the keys
            ambiguous_counter += 1
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
        "ambiguous_rows_in_final_collection": ambiguous_counter,
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