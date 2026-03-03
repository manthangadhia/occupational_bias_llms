"""
Given a data and results directory Path, load all the files as pandas dataframes and return a dictionary of dataframes.
"""

from pathlib import Path
import pandas as pd

def load_json_data(results_dir: Path, file_name_keyword: str = None) -> dict:
    """
    Load all the json files in the results directory as pandas dataframes and return a dictionary of dataframes.

    Args:
        results_dir (Path): Path to the results directory.
        file_name_keyword (str, optional): Keyword to filter the files to be loaded. Defaults to None.

    Returns:
        dict: A dictionary of dataframes with keys as file names and values as dataframes.
    """
    data_frames = {}

    # Load results files
    for file in results_dir.glob('*.json'):
        if file_name_keyword:
            if file_name_keyword in file.name:
                data_frames[file.stem] = pd.read_json(file)
        else:
            data_frames[file.stem] = pd.read_json(file)

    return data_frames