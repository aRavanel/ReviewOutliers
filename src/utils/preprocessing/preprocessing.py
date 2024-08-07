import pandas as pd

# module imports
from src.utils.preprocessing.preprocessing_encoding import encode_data
from src.utils.preprocessing.preprocessing_cleaning import clean_enrich
from src.config import logger


# ==========================================================================
# Exported functions
# ==========================================================================
def merge_dataframes(df_metadata: pd.DataFrame, df_review: pd.DataFrame, max_samples: int = 10_000) -> pd.DataFrame:
    """
    Merges two pandas DataFrames, `df_metadata` and `df_review`, based on the 'parent_asin' column.
    The function takes in the two DataFrames, `df_metadata` and `df_review`, and an optional
    parameter `max_samples` which specifies the maximum number of samples to return.

    Parameters:
    - df_metadata (pd.DataFrame): The first DataFrame to merge.
    - df_review (pd.DataFrame): The second DataFrame to merge.
    - max_samples (int, optional): The maximum number of samples to return. Default is 10,000.

    Returns:
    - pd.DataFrame: The merged DataFrame.
    """

    logger.debug("calling merge_dataframes")

    # Validate inputs
    if df_metadata.empty or df_review.empty:
        raise ValueError("Input DataFrames should not be empty.")

    if "parent_asin" not in df_metadata.columns or "parent_asin" not in df_review.columns:
        raise ValueError("Both DataFrames must contain the 'parent_asin' column.")

    # Merge the datasets on 'parent_asin' with suffixes for duplicate columns
    merged_df = pd.merge(
        df_review, df_metadata, on="parent_asin", how="inner", suffixes=("_review", "_metadata"), validate="many_to_one"
    )
    merged_df = merged_df.dropna(subset=["asin", "parent_asin"])

    # limit to wanted sample size, random_state for reproducibility
    merged_df = merged_df.sample(n=max_samples, random_state=42)

    return merged_df


def preprocess_data(merged_df: pd.DataFrame, training: bool = True, saved_name: str = "") -> pd.DataFrame:
    """
    clean, enrich and encode the data
    """

    logger.debug("calling preprocess_data")

    merged_df = clean_enrich(merged_df, saved_name=saved_name)
    combined_df = encode_data(merged_df, training=training)

    return combined_df
