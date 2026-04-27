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

if __name__ == "__main__":
    import pyarrow.parquet as pq
    schema = pq.read_schema(dolci_professions_file)
    print(schema)