import pandas as pd
from typing import List


def drop_duplicates_with_unhashable(df: pd.DataFrame, ignore_columns: List[str] = None) -> pd.DataFrame:
    """
    Drops duplicates from the DataFrame considering unhashable columns.

    Parameters:
    - df: pd.DataFrame: The input DataFrame.
    - ignore_columns: List[str]: List of columns to ignore when dropping duplicates.

    Returns:
    - pd.DataFrame: The DataFrame with duplicates removed.
    """
    if ignore_columns is None:
        ignore_columns = []

    # Function to detect unhashable columns
    def is_unhashable(series: pd.Series) -> bool:
        try:
            # Drop NaN values
            non_null_series = series.dropna()
            if non_null_series.empty:
                return False  # If the series is empty after dropping NaNs, consider it hashable
            # Try to hash the first non-null value
            hash(tuple(non_null_series.iloc[0]))
            return False
        except TypeError:
            return True

    # Automatically detect columns with unhashable types
    unhashable_columns = [col for col in df.columns if is_unhashable(df[col])]

    # Convert unhashable columns to tuples to make them hashable
    for col in unhashable_columns:
        df[col] = df[col].apply(
            lambda x: tuple(x.items()) if isinstance(x, dict) else tuple(x) if isinstance(x, list) else x
        )

    # Create a subset of columns to consider for duplicates
    subset_columns = [col for col in df.columns if col not in ignore_columns]

    # Drop duplicates considering only the specified subset of columns
    df.drop_duplicates(subset=subset_columns, inplace=True)

    # Convert tuple columns back to their original types if needed
    for col in unhashable_columns:
        df[col] = df[col].apply(
            lambda x: (
                dict(x)
                if isinstance(x, tuple) and all(isinstance(i, tuple) and len(i) == 2 for i in x)
                else list(x) if isinstance(x, tuple) else x
            )
        )

    return df
