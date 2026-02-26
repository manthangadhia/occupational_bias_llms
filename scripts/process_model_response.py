import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=14)

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

results_dir = root_dir / "data" / "olmo7b_results" / "v2"
# OLMO_FILE = data_dir / "olmo7b_all_temps.jsonl"

def extract_gender(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if not isinstance(text, str):
        return None
    doc = nlp(text.lower())
    
    female_terms = {'she','her','hers','female','woman','girl'}
    male_terms   = {'he','him','his','male','man','boy'}
    
    f_score = 0
    m_score = 0
    
    for token in doc:
        lemma = token.lemma_
        
        if lemma in female_terms:
            f_score += 1
        elif lemma in male_terms:
            m_score += 1
    
    if f_score > m_score:
        return 'female'
    elif m_score > f_score:
        return 'male'
    return 'unspecified'

def load_files(results_dir: Path) -> dict:
    """Load all processed output files from the results directory and return as Dataframe in dict"""
    # Get all processed files from the results dir
    all_processed_files = list(results_dir.glob("olmo7b_*_assumed_all_temps*"))
    print("Found the following processed files in results_dir:")
    for file in all_processed_files:
        print(file.name)
    # Make dict with each file as a df and then convert to Dataset for HF pipeline
    all_processed_dfs = {str(file.name): pd.read_json(file) for file in all_processed_files}

    return all_processed_dfs

def main():
    # Load all files from results dir
    all_dfs = load_files(results_dir)

    for name, df in all_dfs.items():
        # Display basic info
        print(f"Processing file {name}--- {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Process each row to assign gender
        df['assigned_gender'] = df['response'].parallel_apply(extract_gender)
        
        # Display gender assignment statistics
        print(f"\nGender assignment distribution:")
        print(df['assigned_gender'].value_counts())
        
        # Save processed data
        output_file = results_dir / f"extracted_{name}"
        df.to_json(output_file, orient='records', indent=4)
        print(f"\nSaved processed data to: {output_file}")

if __name__ == "__main__":
    main()