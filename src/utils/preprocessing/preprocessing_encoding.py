import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# ==========================================================================
# Utils functions
# ==========================================================================


def save_encoder(encoder, file_name):
    """Save the encoder as a pickle file."""
    with open(file_name, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


def load_encoder(file_name):
    """Load the encoder from a pickle file."""
    with open(file_name, "rb") as encoder_file:
        return pickle.load(encoder_file)


# ==========================================================================
# Exported functions
# ==========================================================================


def encode_numerical(merged_df, save_encoders=False, load_encoders=False, encoder_path=None):
    """Scale numerical data: remove the mean and scale to unit variance."""
    numerical_features = merged_df.select_dtypes(include=["number"]).columns.tolist()

    if load_encoders and encoder_path:
        scaler_numerical = load_encoder(os.path.join(encoder_path, "scaler_numerical.pkl"))
    else:
        scaler_numerical = StandardScaler()
        X_numerical = merged_df[numerical_features].values
        scaler_numerical.fit(X_numerical)
        if save_encoders and encoder_path:
            save_encoder(scaler_numerical, os.path.join(encoder_path, "scaler_numerical.pkl"))

    X_numerical_standardized = scaler_numerical.transform(merged_df[numerical_features].values)
    return X_numerical_standardized


def encode_categorical(merged_df: pd.DataFrame, save_encoders=False, load_encoders=False, encoder_path=None):
    """Encode categorical features and scale them to have unit variance."""
    categorical_features = merged_df.select_dtypes(include=["category", "bool"]).columns.tolist()
    X_categorical_scaled = []

    if load_encoders and encoder_path:
        # Load all encoders
        for cat_feature in categorical_features:
            encoder = load_encoder(os.path.join(encoder_path, f"{cat_feature}_encoder.pkl"))
            encoded_data = encoder.transform(merged_df[[cat_feature]])
            num_categories = encoded_data.shape[1]
            scaled_data = encoded_data / num_categories
            X_categorical_scaled.append(scaled_data)
    else:
        # Train and save all encoders
        for cat_feature in categorical_features:
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(merged_df[[cat_feature]])
            encoded_data = encoder.transform(merged_df[[cat_feature]])
            num_categories = encoded_data.shape[1]
            scaled_data = encoded_data / num_categories
            X_categorical_scaled.append(scaled_data)
            if save_encoders and encoder_path:
                save_encoder(encoder, os.path.join(encoder_path, f"{cat_feature}_encoder.pkl"))

    return np.hstack(X_categorical_scaled)


def encode_textual(merged_df: pd.DataFrame, model, save_encoders=False, load_encoders=False, encoder_path=None):
    """Encode textual features using a model and scale the embeddings."""
    if load_encoders and encoder_path:
        review_embeddings = load_encoder(os.path.join(encoder_path, "review_embeddings.pkl"))
        metadata_embeddings = load_encoder(os.path.join(encoder_path, "metadata_embeddings.pkl"))
    else:
        review_embeddings = model.encode(merged_df["review_text"].tolist(), show_progress_bar=True)
        metadata_embeddings = model.encode(merged_df["metadata_text"].tolist(), show_progress_bar=True)

        if save_encoders and encoder_path:
            save_encoder(review_embeddings, os.path.join(encoder_path, "review_embeddings.pkl"))
            save_encoder(metadata_embeddings, os.path.join(encoder_path, "metadata_embeddings.pkl"))

    embedding_size = model.get_sentence_embedding_dimension()
    review_embeddings_scaled = review_embeddings / embedding_size
    metadata_embeddings_scaled = metadata_embeddings / embedding_size

    return np.hstack((review_embeddings_scaled, metadata_embeddings_scaled))


def encode_data(merged_df, model, save_encoders=False, load_encoders=False, encoder_path=None):
    """
    Encode numerical, categorical, and textual data
    and combine all features into a single dataset.
    """
    X_numerical_standardized = encode_numerical(merged_df, save_encoders, load_encoders, encoder_path)
    X_categorical_scaled = encode_categorical(merged_df, save_encoders, load_encoders, encoder_path)
    X_textual_scaled = encode_textual(merged_df, model, save_encoders, load_encoders, encoder_path)

    # Combine all features into a single dataset
    X_combined = np.hstack((X_numerical_standardized, X_categorical_scaled, X_textual_scaled))
    combined_df = pd.DataFrame(X_combined, columns=[f"feature_{i}" for i in range(X_combined.shape[1])])

    return combined_df
