import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade, gunning_fog, flesch_reading_ease
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# module imports
from src.config import logger

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

# download resources
nltk.download("vader_lexicon")

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
    else:  # inference
        scaler_numerical = load_encoder(MODEL_PATH_STANDARDIZER)
        if scaler_numerical is None:
            raise ValueError("Failed to load the numerical scaler.")

    x_numerical_standardized = scaler_numerical.transform(merged_df[numerical_features].values)
    return x_numerical_standardized


def encode_categorical(merged_df: pd.DataFrame, training: bool = True) -> np.ndarray:
    """
    Encode categorical features
    and scale them: so that they are not dominant when computing distances
    """
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

    return np.hstack(x_categorical_scaled)


# def encode_textual(merged_df: pd.DataFrame):
#     """
#     Encode textual features using a model and scale the embeddings.
#     """

#     # compute embeddings
#     review_embeddings = model_embeddings.encode(merged_df["text_review"].tolist(), show_progress_bar=True)
#     metadata_embeddings = model_embeddings.encode(merged_df["text_metadata"].tolist(), show_progress_bar=True)

#     # VADER SENTIMENT SCORE
#     sid = SentimentIntensityAnalyzer()
#     merged_df["sentiment_score"] = merged_df["text_review"].apply(lambda d: sid.polarity_scores(d)["compound"])

#     # sentiment embeddings + cosing similarity
#     sentiment_phrases = {
#         "good": "Good product. I am happy",
#         "bad": "I will never buy this product again. Bad Quality",
#         "expensive": "Very expensive.",
#         "scam": "This is a scam. Do not buy.",
#         "error": "There was an error in the product. The delivery had an issue. Wrong product.",
#     }
#     sentiment_embeddings = {k: model_embeddings.encode([v], show_progress_bar=False) for k, v in sentiment_phrases.items()}
#     sentiment_similarities = {
#         k: np.dot(review_embeddings, v.T).flatten().reshape(-1, 1) for k, v in sentiment_embeddings.items()
#     }

#     # Compute readability scores
#     readability_scores = {
#         "flesch_kincaid": merged_df["text_review"].apply(flesch_kincaid_grade).values.reshape(-1, 1),
#         "gunning_fog": merged_df["text_review"].apply(gunning_fog).values.reshape(-1, 1),
#         "flesch_reading_ease": merged_df["text_review"].apply(flesch_reading_ease).values.reshape(-1, 1),
#     }

#     # Length-based features
#     length_features = {
#         "length_char": merged_df["text_review"].apply(len).values.reshape(-1, 1),
#         "length_word": merged_df["text_review"].apply(lambda x: len(x.split())).values.reshape(-1, 1),
#         "length_sentence": merged_df["text_review"].apply(lambda x: len(x.split("."))).values.reshape(-1, 1),
#     }

#     # Interaction features
#     interaction_scores = np.dot(review_embeddings, metadata_embeddings.T).diagonal().reshape(-1, 1)

#     # embeddings tuple
#     # embedding_size = model_embeddings.get_sentence_embedding_dimension()
#     # review_embeddings_scaled = review_embeddings / embedding_size
#     # metadata_embeddings_scaled = metadata_embeddings / embedding_size
#     # specific_features = np.hstack((good_similarity, expensive_similarity, scam_similarity, error_similarity))
#     # specific_features = specific_features / len(specific_features)
#     # embeddings = (review_embeddings_scaled, metadata_embeddings_scaled, specific_features)
#     # return np.hstack(embeddings)

#     # Concatenate all features
#     all_features = np.hstack(
#         list(sentiment_similarities.values())
#         + list(readability_scores.values())
#         + list(length_features.values())
#         + [interaction_scores, merged_df["sentiment_score"].values.reshape(-1, 1)]
#     )

#     return all_features


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

    x_numerical_standardized = encode_numerical(merged_df, training=training)
    x_categorical_scaled = encode_categorical(merged_df, training=training)
    x_combined = np.hstack((x_numerical_standardized, x_categorical_scaled))
    combined_df = pd.DataFrame(x_combined, columns=[f"feature_{i}" for i in range(x_combined.shape[1])])

    return combined_df
