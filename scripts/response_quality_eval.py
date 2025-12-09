import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any
import re
from pathlib import Path
import sys

from diversity import compression_ratio, homogeneization_score, ngram_diversity_score
import nltk
from nltk.tokenize import word_tokenize

# -------------------------
# configuration
# -------------------------

# Add utils to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

data_dir = root_dir / "data"
models_dir = root_dir / "models"
models_dir.mkdir(exist_ok=True)

PROMPT_FILE = data_dir / "prompts.txt"
RESULTS_FILE = data_dir / "results.jsonl"
# -------------------------

class ResponseEvaluator:
    """Evaluate quality and diversity of LLM responses."""
    
    def __init__(self, responses_file: str):
        """
        Args:
            responses_file: Path to JSONL file with format:
                {"prompt": "...", "model": "...", "response": "..."}
        """
        self.responses = self._load_responses(responses_file)
        
    def _load_responses(self, file_path: str) -> List[Dict[str, Any]]:
        """Load responses from JSONL file."""
        responses = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                responses.append(json.loads(line))
        return responses
    

    def tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return word_tokenize(text.lower())
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    

    def distinct_n(self, responses: List[str], n: int) -> float:
        """
        Calculate Distinct-n score across all responses.
        Higher values indicate more diversity.
        """
        ngd = ngram_diversity_score(responses, n)
        return ngd
    
    def self_bleu(self, responses: List[str]) -> float:
        """
        Calculate Self-BLEU score as homogenisation score from `diversity`.
        Lower scores indicate more diversity.
        Each response is compared against all others.
        """
        hs = homogeneization_score(responses, method='self-bleu')
        return hs
    
    
    def repetition_score(self, text: str) -> float:
        """
        Quantify how much repetition is within each response, using compression as a proxy.
        Higher values indicate more repetition.
        """
        cr = compression_ratio(text)
        return cr
    
    
    def response_length_stats(self, responses: List[str]) -> Dict[str, float]:
        """Calculate length statistics for responses."""
        lengths = [len(self.tokenize_simple(r)) for r in responses]
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths)
        }

    
    def evaluate_by_model(self) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics grouped by model."""
        results = {}
        
        # Group responses by model
        model_responses = defaultdict(list)
        for item in self.responses:
            model_responses[item['model']].append(item['response'])
        
        # Calculate metrics for each model
        for model, responses in model_responses.items():
            results[model] = {
                'num_responses': len(responses),
                'distinct_1': self.distinct_n(responses, 1),
                'distinct_2': self.distinct_n(responses, 2),
                'distinct_3': self.distinct_n(responses, 3),
                'self_bleu': self.self_bleu(responses),
                'avg_repetition (compression)': np.mean([self.repetition_score(r) for r in responses]),
                **self.response_length_stats(responses)
            }
        
        return results
    
    def evaluate_overall(self) -> Dict[str, float]:
        """Evaluate metrics across all responses."""
        all_responses = [item['response'] for item in self.responses]
        
        return {
            'num_responses': len(all_responses),
            'distinct_1': self.distinct_n(all_responses, 1),
            'distinct_2': self.distinct_n(all_responses, 2),
            'distinct_3': self.distinct_n(all_responses, 3),
            'self_bleu': self.self_bleu(all_responses),
            'avg_repetition (compression)': np.mean([self.repetition_score(r) for r in all_responses]),
            **self.response_length_stats(all_responses)
        }
    
    def print_report(self):
        """Print a formatted evaluation report."""
        print("=" * 60)
        print("OVERALL EVALUATION")
        print("=" * 60)
        
        overall = self.evaluate_overall()
        for metric, value in overall.items():
            print(f"{metric:.<30} {value:.4f}")
        
        print("\n" + "=" * 60)
        print("EVALUATION BY MODEL")
        print("=" * 60)
        
        by_model = self.evaluate_by_model()
        for model, metrics in by_model.items():
            print(f"\n{model}:")
            print("-" * 60)
            for metric, value in metrics.items():
                print(f"  {metric:.<28} {value:.4f}")
    
    def save_results(self, output_file: str):
        """Save evaluation results to JSON file."""
        results = {
            'overall': self.evaluate_overall(),
            'by_model': self.evaluate_by_model()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    evaluator = ResponseEvaluator(RESULTS_FILE)
    evaluator.print_report()
    evaluator.save_results(data_dir / "evaluation_results.json")