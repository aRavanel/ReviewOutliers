import gzip
import json
import os
from io import StringIO
from tqdm import tqdm
import pandas as pd

# module imports
from logger_config import logger

# ==========================================================================
# Utils functions
# ==========================================================================


# ==========================================================================
# Exported functions
# ==========================================================================


def read_json_lines(file_path: str, max_samples: int = 10_000) -> pd.DataFrame:
    """
    Function to read JSON Lines files and convert to DataFrame
    When data is in  JSON Lines format (.jsonl), each line is a separate JSON object.
    -> should read the file line by line and parse each line as a JSON object, then concatenate them into a DataFrame.
    else get ValueError: Trailing data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(tqdm(file, total=max_samples, desc="Reading lines")):
            if i >= max_samples:
                break
            data.append(pd.read_json(StringIO(line), lines=True))
    return pd.concat(data, ignore_index=True)


def decompress_to_json(gz_filename, json_path):
    """
    Decompress a .gz file containing JSON lines and save it as a .json
    Dumps all at once into a json. if memory issue could be done as a jsonl file, one line at a time
    """
    # read lines one by one
    json_list = []
    with gzip.open(gz_filename, "rt", encoding="utf-8") as gz_file:
        for line in gz_file:
            json_line = json.loads(line)
            json_list.append(json_line)
    # dumps it into json file
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_list, json_file, indent=4)
    print(f"Decompressed and saved to {json_path}")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load data into a pandas DataFrame.
    """
    logger.debug("calling load_dataframe")

    try:
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        raise
