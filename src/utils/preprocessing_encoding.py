import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================================================
# Utils functions
# ==========================================================================


# ==========================================================================
# Exported functions
# ==========================================================================

def encode_numerical(merged_df) :
    """"""
    # Directly scale numerical data : remove the mean and scale to unit variance
    numerical_features = merged_df.select_dtypes(include=['number']).columns.tolist()
    scaler_numerical = StandardScaler()
    X_numerical = merged_df[numerical_features].values
    X_numerical_standardized = scaler_numerical.fit_transform(X_numerical)
    return X_numerical_standardized


def encode_categorical(merged_df: pd.DataFrame) :
    """
    """
    # Scale Encoded Categorical Features
    # After encoding (ex for OH), scale each group of categorical features so that the entire group has unit variance.
    # Note : 
    # - TODO could use different scaler if too big number of categories
    # - if sparse use hstack of scipy, else numpy
    # - X_categorical_scaled is a list of arrays -> need to put it in a single array
    # TODO: save the encoders for future use (inference)
    categorical_features = merged_df.select_dtypes(include=['category', 'bool']).columns.tolist()
    
    categorical_encoder_list = []
    X_categorical_scaled = []
    for cat_feature in categorical_features:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(merged_df[[cat_feature]])
        num_categories = encoded_data.shape[1]
        scaled_data = encoded_data / num_categories
        X_categorical_scaled.append(scaled_data)
        categorical_encoder_list.append(encoder)
        
    return np.hstack(X_categorical_scaled)


def encode_textual(merged_df: pd.DataFrame, model):
    """
    """
    # embeddings of textual features
    review_embeddings = model.encode(merged_df['review_text'].tolist(), show_progress_bar=True)
    metadata_embeddings = model.encode(merged_df['metadata_text'].tolist(), show_progress_bar=True)

    # Scale Text Embeddings: 
    # - possibility A:Ensure the text embeddings have unit variance as a group. + weighting
    # - possibility B:if already in [-1, 1], just do weighting to keep hyperspace geometry
    embedding_size = model.get_sentence_embedding_dimension()
    review_embeddings_scaled = review_embeddings/embedding_size
    metadata_embeddings_scaled = metadata_embeddings/embedding_size

    # Other solution : add some similarity score such as
    # - positive_review
    # - similarity to description
    # - language
    # ...
    
    return np.hstack((review_embeddings_scaled, metadata_embeddings_scaled))

