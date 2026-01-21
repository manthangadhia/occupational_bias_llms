import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import re
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=4)

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

data_dir = root_dir / "data" / "olmo7b_results"
OLMO_FILE = data_dir / "olmo7b_all_temps.jsonl"

def remove_preamble(text):
    pattern = r'.*?(?:assistant\s*)?.*?Assuming the individual is\s*(.*)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def assign_gender(model_response):
    response = model_response.lower()
    first_few_words = ' '.join(response.split()[:20])
    female_indicators = ['female', 'she', 'her', 'woman']
    male_indicators = [ 'male', 'he', 'him', 'man']   
    unassigned = ['gender', '[gender]']
    for indicator in female_indicators:
        if indicator in first_few_words:
            return 'female'
    for indicator in male_indicators:
        if indicator in first_few_words:
            return 'male'
    for indicator in unassigned:
        if indicator in first_few_words:
            return 'gender'
    return None

def main():
    # Load the data
    olmo_df = pd.read_json(OLMO_FILE, lines=True)

    # Display basic info
    print(f"Shape: {olmo_df.shape}")
    print(f"\nColumns: {olmo_df.columns.tolist()}")

    # Process each row to replace response with clipped response
    olmo_df['response'] = olmo_df['response'].parallel_apply(remove_preamble)

    # Process each row to assign gender
    olmo_df['assigned_gender'] = olmo_df['response'].parallel_apply(assign_gender)
    
    # Display gender assignment statistics
    print(f"\nGender assignment distribution:")
    print(olmo_df['assigned_gender'].value_counts())
    print(f"\nNull assignments: {olmo_df['assigned_gender'].isnull().sum()}")
    
    # Save processed data
    output_file = data_dir / "olmo7b_all_temps_processed.jsonl"
    olmo_df.to_json(output_file, orient='records', lines=True)
    print(f"\nSaved processed data to: {output_file}")
    
    return olmo_df

if __name__ == "__main__":
    df = main()