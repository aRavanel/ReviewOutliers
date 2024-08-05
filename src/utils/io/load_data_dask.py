import dask.dataframe as dd
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataframe_dask(
    file_path: str, sample: int = None, filter_expr: dict = None, chunksize: int = 10000
) -> dd.DataFrame:
    """
    Load data into a dask DataFrame, with options to load a sample of the data or filter rows.

    Parameters:
    - file_path (str): The path to the file to be loaded.
    - sample (int, optional): The number of samples to load from the dataset. If None, load the entire dataset.
    - filter_expr (dict, optional): A dictionary to filter rows. Keys are column names and values are filter values.
    - chunksize (int, optional): The number of rows to read at a time (only for CSV and JSON files).

    Returns:
    - dd.DataFrame: The loaded Dask DataFrame.
    """
    logger.debug("calling load_dataframe")
    logger.info(f"Current working directory: {os.getcwd()}")

    try:
        if file_path.endswith(".parquet"):
            # Read Parquet file with optional filtering
            df = dd.read_parquet(file_path)

        elif file_path.endswith(".csv"):
            # Read CSV file in chunks
            df = dd.read_csv(file_path, blocksize=chunksize)

        elif file_path.endswith(".json"):
            # Read JSON file in chunks
            df = dd.read_json(file_path, blocksize=chunksize, lines=True)

        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            # Dask doesn't support pickle directly
            df = pd.read_pickle(file_path)
            if sample is not None:
                df = df.sample(n=sample)
            return df

        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if filter_expr:
            for column, value in filter_expr.items():
                df = df[df[column] == value]

        if sample is not None:
            df = df.sample(n=sample)

        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        raise
