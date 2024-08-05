import pandas as pd
import os
import logging
import pyarrow.parquet as pq
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


def load_dataframe(file_path: str, sample: int = None, filter_expr: str = None, chunksize: int = 10000) -> pd.DataFrame:
    """
    Load data into a pandas DataFrame, with options to load a sample of the data or filter rows.

    Parameters:
    - file_path (str): The path to the file to be loaded.
    - sample (int, optional): The number of samples to load from the dataset. If None, load the entire dataset.
    - filter_expr (str, optional): An expression to filter rows (only for Parquet files).
    - chunksize (int, optional): The number of rows to read at a time (only for CSV and JSON files).

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    logger.debug("calling load_dataframe")
    logger.info(f"Current working directory: {os.getcwd()}")

    try:
        if file_path.endswith(".parquet"):
            # Read Parquet file with optional filtering
            if filter_expr:
                dataset = ds.dataset(file_path, format="parquet")
                table = dataset.to_table(filter=ds.field(filter_expr))
                df = table.to_pandas()
            else:
                df = pd.read_parquet(file_path)
            if sample is not None:
                df = df.sample(n=sample)
            return df

        elif file_path.endswith(".csv"):
            # Read CSV file in chunks
            chunk_list = []
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk_list.append(chunk)
                if sample is not None and len(chunk_list) * chunksize >= sample:
                    break
            df = pd.concat(chunk_list, ignore_index=True)
            if sample is not None:
                df = df.sample(n=sample)
            return df

        elif file_path.endswith(".json"):
            # Read JSON file in chunks
            chunk_list = []
            for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
                chunk_list.append(chunk)
                if sample is not None and len(chunk_list) * chunksize >= sample:
                    break
            df = pd.concat(chunk_list, ignore_index=True)
            if sample is not None:
                df = df.sample(n=sample)
            return df

        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            df = pd.read_pickle(file_path)
            if sample is not None:
                df = df.sample(n=sample)
            return df

        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        raise
