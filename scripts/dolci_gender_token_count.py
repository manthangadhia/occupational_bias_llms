from pathlib import Path
import json
from multiprocessing import Pool
# -------------------------
# Configuration
# -------------------------
root_dir = Path(__file__).parent.parent.parent                  # .py < scripts < occ_bias < root > data         # structure on euler
project_dir = root_dir / "occ_bias"
dolci_dir = project_dir / "data" / "dolci_sft"
dolci_dataset = dolci_dir / "dolci_sft.parquet"                 # full, original dolci-sft dataset
gender_tokens_dolci_output = dolci_dir / "gender_tokens_dolci_counts.json"

import datasets
import ahocorasick
from tqdm import tqdm

def load_dolci_data(data_path: Path) -> datasets.Dataset:
    """Load the Dolci-SFT dataset from the given parquet file path."""
    return datasets.load_dataset("parquet", data_files=str(data_path))["train"]

from collections import Counter
import re

def whole_word(text, start_index, end_index):
    """
    Given the target word and the character-sequence hit, 
    ensure that the hit is a whole word and not a subword 
    (like 'teacher' for 'her')
    """
    # Check if the character before the match is not a word character
    if start_index > 0 and re.match(r'\w', text[start_index - 1]):
        return False
    # Check if the character after the match is not a word character
    if end_index + 1 < len(text) and re.match(r'\w', text[end_index + 1]):
        return False
    return True

def search_and_count(text, automaton):
    """
    Search the text for matches using the Aho-Corasick automaton and count occurrences.
    """
    counts = Counter()
    for end_index, word in automaton.iter(text):
        start_index = end_index - len(word) + 1
        if whole_word(text, start_index, end_index):
            counts[word] += 1
    return counts

def build_automaton(word_list):
    automaton = ahocorasick.Automaton()
    for word in word_list:
        automaton.add_word(word, word)
    automaton.make_automaton()
    return automaton

def count_gendered_words(args):
    data_path, word_list, label, position = args
    automaton = build_automaton(word_list)
    dataset = load_dolci_data(data_path)
    total = 0
    for row in tqdm(
        dataset,
        desc=f"{label} count",
        position=position,
        leave=True,
        dynamic_ncols=True,
    ):
        all_content = " ".join(
            t["content"] for t in row["messages"] if t["content"]
        ).lower()
        if not all_content:
            continue
        total += sum(search_and_count(all_content, automaton).values())
    return total

def main():
    # gendered words
    male = [
        # Pronouns & Articles
        "he", "him", "his", "himself",
        
        # Core Nouns
        "boy", "boys", "man", "men", "guy", "guys", "dude", "dudes", "lad", "lads",
        "gentleman", "gentlemen", "male", "males",
        
        # Familial Roles
        "father", "fathers", "dad", "dads", "daddy", "daddies", "pa", "pop",
        "son", "sons", "brother", "brothers", "bro", "bros",
        "uncle", "uncles", "nephew", "nephews",
        "grandfather", "grandfathers", "grandpa", "grandpas", "grandson", "grandsons",
        "husband", "husbands", "fiance", "fiances", "widower", "widowers",
        "paternal", "patriarch", "patriarchs",
        
        # Titles & Honorifics
        "mr", "mr.", "mister", "sir", "sirs", "lord", "lords", "king", "kings",
    ]

    female = [
        # Pronouns & Articles
        "she", "her", "hers", "herself",
        
        # Core Nouns
        "girl", "girls", "woman", "women", "gal", "gals", "lady", "ladies",
        "gentlewoman", "gentlewomen", "female", "females",
        
        # Familial Roles
        "mother", "mothers", "mom", "moms", "mommy", "mommies", "ma", "mama",
        "daughter", "daughters", "sister", "sisters", "sis",
        "aunt", "aunts", "niece", "nieces",
        "grandmother", "grandmothers", "grandma", "grandmas", "granddaughter", "granddaughters",
        "wife", "wives", "fiancee", "fiancees", "widow", "widows",
        "maternal", "matriarch", "matriarchs",
        
        # Titles & Honorifics
        "ms", "ms.", "mrs", "mrs.", "miss", "madam", "madame", "ma'am", "queen", "queens",
    ]

    with Pool(2) as pool:
        total_male_count, total_female_count = pool.map(
            count_gendered_words,
            [
                (dolci_dataset, male, "Male", 0),
                (dolci_dataset, female, "Female", 1),
            ]
        )

    print(f"Total male gendered words found: {total_male_count}")
    print(f"Total female gendered words found: {total_female_count}")

    # Save the results to a JSON file
    with open(gender_tokens_dolci_output, "w") as f:
        json.dump({
            "total_male_count": total_male_count,
            "total_female_count": total_female_count
        }, f)

if __name__ == "__main__":
    main()