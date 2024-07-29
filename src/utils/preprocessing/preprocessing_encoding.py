import os
from typing import Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
import pickle

# ==========================================================================
# Module variables
# ==========================================================================
from src.config import (
    MODEL_NAME_EMBEDDINGS,
    MODEL_PATH_STANDARDIZER,
    BASE_PATH_MODEL,
)

# pickled models
model_embeddings = SentenceTransformer(MODEL_NAME_EMBEDDINGS)  # text embeddings

# ==========================================================================
# Utils functions
# ==========================================================================


def save_encoder(encoder, file_name: str) -> None:
    """Save the encoder as a pickle file."""
    with open(file_name, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


def load_encoder(file_name: str) -> Any:
    """Load the encoder from a pickle file."""
    with open(file_name, "rb") as encoder_file:
        return pickle.load(encoder_file)


def encode_numerical(merged_df: pd.DataFrame, training: bool = True):
    """
    Scale numerical data: remove the mean and scale to unit variance.
    """
    numerical_features = merged_df.select_dtypes(include=["number"]).columns.tolist()
    if training:
        scaler_numerical = StandardScaler()
        x_numerical = merged_df[numerical_features].values
        scaler_numerical.fit(x_numerical)
        save_encoder(scaler_numerical, MODEL_PATH_STANDARDIZER)
    else:
        # inference
        scaler_numerical = load_encoder(MODEL_PATH_STANDARDIZER)

    x_numerical_standardized = scaler_numerical.transform(merged_df[numerical_features].values)
    return x_numerical_standardized


def encode_categorical(merged_df: pd.DataFrame, training: bool = True):
    """
    Encode categorical features and scale them to have unit variance.
    """
    categorical_features = merged_df.select_dtypes(include=["category", "bool"]).columns.tolist()
    X_categorical_scaled = []

    if training:
        # Train and save all encoders
        for cat_feature in categorical_features:
            # encode
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(merged_df[[cat_feature]])
            save_encoder(encoder, os.path.join(BASE_PATH_MODEL, f"{cat_feature}_encoder.pkl"))
            encoded_data = encoder.transform(merged_df[[cat_feature]])

            # scale
            num_categories = encoded_data.shape[1]
            scaled_data = encoded_data / num_categories
            X_categorical_scaled.append(scaled_data)

    else:
        # Load all encoders
        for cat_feature in categorical_features:
            encoder = load_encoder(os.path.join(BASE_PATH_MODEL, f"{cat_feature}_encoder.pkl"))
            encoded_data = encoder.transform(merged_df[[cat_feature]])
            num_categories = encoded_data.shape[1]
            scaled_data = encoded_data / num_categories
            X_categorical_scaled.append(scaled_data)

    return np.hstack(X_categorical_scaled)


def encode_textual(merged_df: pd.DataFrame):
    """
    Encode textual features using a model and scale the embeddings.
    """

    # compute embeddings
    review_embeddings = model_embeddings.encode(merged_df["review_text"].tolist(), show_progress_bar=True)
    metadata_embeddings = model_embeddings.encode(merged_df["metadata_text"].tolist(), show_progress_bar=True)

    # scale the embeddings
    embedding_size = model_embeddings.get_sentence_embedding_dimension()
    review_embeddings_scaled = review_embeddings / embedding_size
    metadata_embeddings_scaled = metadata_embeddings / embedding_size

    # add some specific features
    good_embeddings = model_embeddings.encode(["Good product. I am happy"], show_progress_bar=False)
    expensive_embeddings = model_embeddings.encode(["Very expensive."], show_progress_bar=False)
    scam_embeddings = model_embeddings.encode(["This is a scam. Do not buy."], show_progress_bar=False)
    error_embeddings = model_embeddings.encode(
        ["There was an error in the product. The delivery had an issue. Wrong product."], show_progress_bar=False
    )

    # compute cosine similarity of review with the specific features above
    good_similarity = np.dot(review_embeddings_scaled, good_embeddings.T)
    expensive_similarity = np.dot(review_embeddings_scaled, expensive_embeddings.T)
    scam_similarity = np.dot(review_embeddings_scaled, scam_embeddings.T)
    error_similarity = np.dot(review_embeddings_scaled, error_embeddings.T)
    specific_features = np.hstack((good_similarity, expensive_similarity, scam_similarity, error_similarity))
    specific_features = specific_features / len(specific_features)

    # embeddings tuple
    embeddings = (review_embeddings_scaled, metadata_embeddings_scaled, specific_features)

    return np.hstack(embeddings)


# ==========================================================================
# Exported functions
# ==========================================================================


def encode_data(merged_df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Encode numerical, categorical, and textual data
    and combine all features into a single dataset.
    """
    x_numerical_standardized = encode_numerical(merged_df, training)
    x_categorical_scaled = encode_categorical(merged_df, training)
    x_textual_scaled = encode_textual(merged_df)

    # Combine all features into a single dataset
    x_combined = np.hstack((x_numerical_standardized, x_categorical_scaled, x_textual_scaled))
    combined_df = pd.DataFrame(x_combined, columns=[f"feature_{i}" for i in range(x_combined.shape[1])])

    return combined_df
