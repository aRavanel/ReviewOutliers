import os
import pickle
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================================================
# Module variables
# ==========================================================================
from src.config import (
    logger,
    MODEL_PATH_STANDARDIZER,
    BASE_PATH_MODEL,
)

# ==========================================================================
# Utils functions
# ==========================================================================


def save_encoder(encoder: Any, file_name: str) -> None:
    """Save the encoder as a pickle file."""
    try:
        with open(file_name, "wb") as encoder_file:
            pickle.dump(encoder, encoder_file)
    except IOError as e:
        logger.error(f"Error saving encoder to {file_name}: {e}")


def load_encoder(file_name: str) -> Any:
    """Load the encoder from a pickle file."""
    try:
        with open(file_name, "rb") as encoder_file:
            return pickle.load(encoder_file)
    except IOError as e:
        logger.error(f"Error loading encoder from {file_name}: {e}")
        return None


def _diminish_columns_numerical(data: np.ndarray, numerical_features: List[str], column_list: List[str]) -> np.ndarray:
    """Diminish the importance of specified columns in numerical features."""
    for col in column_list:
        if col in numerical_features:
            idx = numerical_features.index(col)
            data[:, idx] /= len(column_list)
    return data


def encode_features(merged_df: pd.DataFrame, training: bool = True) -> np.ndarray:
    """
    Encode both numerical and categorical features.

    Parameters:
    - merged_df: pd.DataFrame: Dataframe containing the features.
    - training: bool: Flag indicating if the encoding is for training or inference.

    Returns:
    - Tuple containing the standardized numerical features and scaled categorical features.
    """
    # Encode numerical features
    numerical_features = merged_df.select_dtypes(include=["number"]).columns.tolist()
    if training:
        scaler_numerical = StandardScaler()
        x_numerical = merged_df[numerical_features].values
        scaler_numerical.fit(x_numerical)
        save_encoder(scaler_numerical, MODEL_PATH_STANDARDIZER)
    else:  # inference
        scaler_numerical = load_encoder(MODEL_PATH_STANDARDIZER)
        if scaler_numerical is None:
            raise ValueError("Failed to load the numerical scaler.")

    x_numerical_standardized = scaler_numerical.transform(merged_df[numerical_features].values)

    # Encode categorical features
    categorical_features = merged_df.select_dtypes(include=["category", "bool"]).columns.tolist()
    x_categorical_scaled = []

    for cat_feature in categorical_features:
        if training:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(merged_df[[cat_feature]])
            save_encoder(encoder, os.path.join(BASE_PATH_MODEL, f"{cat_feature}_encoder.pkl"))
        else:
            encoder = load_encoder(os.path.join(BASE_PATH_MODEL, f"{cat_feature}_encoder.pkl"))
            if encoder is None:
                raise ValueError(f"Failed to load encoder for {cat_feature}.")

        encoded_data = encoder.transform(merged_df[[cat_feature]])
        num_categories = encoded_data.shape[1]

        if isinstance(encoded_data, np.ndarray):
            scaled_data = encoded_data / num_categories
        else:  # sparse matrix case
            scaled_data = encoded_data.toarray() / num_categories

        x_categorical_scaled.append(scaled_data)

    x_categorical_scaled = (
        np.hstack(x_categorical_scaled) if x_categorical_scaled else np.array([]).reshape(len(merged_df), 0)
    )

    # Diminish importance of specified columns in numerical features
    diminish_columns = [
        ["year", "month", "day", "hour"],
        ["sentiment_score", "similarity_good", "similarity_bad", "similarity_expensive"],
        ["similarity_scam", "similarity_error"],
        ["readability", "length_char", "length_word"],
        ["rating", "rating_deviation", "average_rating"],
    ]
    for columns in diminish_columns:
        x_numerical_standardized = _diminish_columns_numerical(x_numerical_standardized, numerical_features, columns)

    return np.hstack((x_numerical_standardized, x_categorical_scaled))


# # ==========================================================================
# # Exported functions
# # ==========================================================================


def encode_data(merged_df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Encode numerical, categorical, and textual data
    and combine all features into a single dataset.

    Parameters:
    - merged_df (pd.DataFrame): DataFrame containing the data to encode.
    - training (bool): If True, fit the encoders and save them; if False, load existing encoders.

    Returns:
    - pd.DataFrame: DataFrame containing the encoded data.
    """
    logger.debug("calling encode_data")
    x_combined = encode_features(merged_df, training=training)
    combined_df = pd.DataFrame(x_combined, columns=[f"feature_{i}" for i in range(x_combined.shape[1])])

    return combined_df
