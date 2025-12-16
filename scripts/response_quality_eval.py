import nltk
from nltk.tokenize import word_tokenize
import json


from pathlib import Path
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"

OUTPUT_FILE = data_dir / "output_ift.jsonl"

def repetition_ratio(response, n=3) -> float:
    """
    Calculate the repetition ratio of words in a response.
    
    Args:
        response (str): The generated response text.
        n (int): n-gram size to consider for repetition; default=3.
        
    Returns:
        float: Ratio of repeated words to total words in response
    """
    tokens = word_tokenize(response.lower())
    
    ngrams = list(nltk.ngrams(tokens, n))
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    rr = 1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
    return rr

if __name__ == "__main__":
    # Dictionary to store results grouped by n-gram size, model, and profile_id
    results = {}
    
    # Load all data once
    data_list = []
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    
    # Loop through different n-gram sizes
    for n in [1, 2, 3, 4]:
        results[n] = {}
        
        for data in data_list:
            profile_id = data['profile_id']
            model = data['model_name']
            response_number = data['response_number']
            response = data['response']
            
            # Create nested dictionary structure if not exists
            if model not in results[n]:
                results[n][model] = {}
            if profile_id not in results[n][model]:
                results[n][model][profile_id] = {}
            
            # Calculate repetition ratio for this response with current n
            rr = repetition_ratio(response, n=n)
            results[n][model][profile_id][response_number] = rr
    
    # Print results organized by n-gram size, model, and profile_id
    for n in [1, 2, 3, 4]:
        print(f"\n{'#' * 80}")
        print(f"N-GRAM SIZE: {n}")
        print(f"{'#' * 80}")
        
        for model, profiles in results[n].items():
            print(f"\nModel: {model}")
            print("=" * 80)
            for profile_id, responses in profiles.items():
                print(f"\n  Profile ID: {profile_id}")
                total_rr = 0
                for response_num in sorted(responses.keys()):
                    rr = responses[response_num]
                    total_rr += rr
                    print(f"    Response {response_num}: {rr:.4f}")
                avg_rr = total_rr / len(responses)
                print(f"    Average: {avg_rr:.4f}")
            