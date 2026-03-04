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
    for file in results_dir.glob('olmo7b*.json'):
        if file_name_keyword:
            if file_name_keyword in file.name:
                data_frames[file.stem] = pd.read_json(file)
        else:
            data_frames[file.stem] = pd.read_json(file)

    return data_frames

def save_dataframes(data_frames: dict, output_dir: Path):
    """
    Save the dataframes in the output directory as json files.

    Args:
        data_frames (dict): A dictionary of dataframes with keys as file names and values as dataframes.
        output_dir (Path): Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, df in data_frames.items():
        df.to_json(output_dir / f"{key}.json", orient='records', indent=4)