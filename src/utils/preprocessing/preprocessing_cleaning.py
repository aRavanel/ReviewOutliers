import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease  # flesch_kincaid_grade, gunning_fog,
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# module imports
from src.config import logger

# ==========================================================================
# Module variables
# ==========================================================================
from src.config import MODEL_NAME_EMBEDDINGS

# pickled models
model_embeddings = SentenceTransformer(MODEL_NAME_EMBEDDINGS, trust_remote_code=True)  # text embeddings

# download resources
nltk.download("vader_lexicon")


# ==========================================================================
# Utils functions
# ==========================================================================
def convert_timestamp(ts):
    """
    Convert a timestamp to a datetime object.

    Args:
        ts (int, str, or pd.Timestamp): The timestamp to convert.
            If the timestamp is an integer, it is treated as milliseconds
            since epoch. If it is a string, it is parsed as a datetime string.
            If it is already a pd.Timestamp object, it is returned as is.

    Returns:
        pd.Timestamp or None: The converted timestamp as a pd.Timestamp object,
            or None if the timestamp could not be converted.
    """
    # Check if the timestamp is missing
    if pd.isnull(ts):
        return None
    # Check if the timestamp is an integer (milliseconds since epoch)
    if isinstance(ts, int):
        # Convert from milliseconds since epoch to datetime
        return pd.to_datetime(ts, unit="ms")
    # Check if the timestamp is a string
    elif isinstance(ts, str):
        # Try to parse the string timestamp
        try:
            return pd.to_datetime(ts)
        except ValueError:
            return None
    # Return the timestamp as is if it is already a pd.Timestamp object
    return ts


def clean_enrich(df_in: pd.DataFrame, saved_name: str = "") -> pd.DataFrame:
    """
    Clean and enrich the input DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame.
        saved_name (str, optional): Path to save enriched DataFrame. Defaults to "".

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    """ """

    logger.debug("calling clean_enrich")

    df = df_in.copy()

    # Expand timestamp
    df["timestamp"] = df["timestamp"].apply(convert_timestamp)
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    if len(df) > 1:
        # User based rating
        average_ratings_user = df.groupby("user_id")["rating"].mean()
        count_ratings_user = df.groupby("user_id")["rating"].count()
        df["average_rating_user"] = df["user_id"].map(average_ratings_user)
        df["count_rating_user"] = df["user_id"].map(count_ratings_user)
        df["rating_deviation"] = abs(df["average_rating"] - df["average_rating_user"])

        # Total number of reviews
        num_reviews = df.groupby("asin").size()
        df["num_reviews"] = df["asin"].map(num_reviews)
    else:
        df["average_rating_user"] = df["rating"]
        df["count_rating_user"] = 1
        df["rating_deviation"] = abs(df["average_rating"] - df["average_rating_user"])
        df["num_reviews"] = 1

    # Handle missing values - categories
    df["store"] = df["store"].fillna("unknown").astype("category")

    # Handle missing values - str
    # concatenate some text together -> this way less features because each embeddings is of high dimensionality
    df["user_id"] = df["user_id"].fillna("").astype("str")

    df["title_review"] = df["title_review"].fillna("").astype("str")
    df["text"] = df["text"].fillna("").astype("str")
    df["text_review"] = df["title_review"] + "/n/n" + df["text"]

    df["title_metadata"] = df["title_metadata"].fillna("").astype("str")
    df["features"] = df["features"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    df["description"] = df["description"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    df["text_metadata"] = df[["title_metadata", "description", "features"]].astype(str).agg("/n/n".join, axis=1)

    # Handle missing values - numeric
    df["average_rating"] = df["average_rating"].fillna(-1).astype("float")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(-1)
    int_fields = {
        "year",
        "month",
        "day",
        "hour",
        "rating",
        "helpful_vote",
        "verified_purchase",
        "rating_number",
    }
    for field in int_fields:
        df[field] = df[field].fillna(-1).astype(int)

    # Enrich textual data with
    # -----------------------

    # Compute embeddings
    logger.debug("computing review embeddings")
    review_embeddings = model_embeddings.encode(df["text_review"].tolist(), show_progress_bar=True)

    logger.debug("computing metadata embeddings")
    metadata_embeddings = model_embeddings.encode(df["text_metadata"].tolist(), show_progress_bar=True)

    # VADER Sentiment Score : from -1 (negative) to 1 (positive)
    sid = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["text_review"].apply(lambda d: sid.polarity_scores(d)["compound"])

    # Sentiment embeddings + cosine similarity
    sentiment_phrases = {
        "good": "Good product. I am happy",
        "bad": "I will never buy this product again. Bad Quality",
        "expensive": "Very expensive.",
        "scam": "This is a scam. Do not buy.",
        "error": "There was an error in the product. The delivery had an issue. Wrong product.",
    }
    sentiment_embeddings = {k: model_embeddings.encode([v], show_progress_bar=False) for k, v in sentiment_phrases.items()}
    for k, v in sentiment_embeddings.items():
        df[f"similarity_{k}"] = np.dot(review_embeddings, v.T).flatten()

    # Compute mean of several readability scores (possibilities : flesch_reading_ease, flesch_kincaid, gunning_fog, ...)
    df["readability"] = df["text_review"].apply(flesch_reading_ease) / 100  # scale 0 - 100 with 100 easy to read

    # Length-based features (possibilities : char, word, sentence)
    df["length_char"] = df["text_review"].apply(len)
    df["length_word"] = df["text_review"].apply(lambda x: len(x.split()))

    # Interaction features
    df["interaction_score"] = np.dot(review_embeddings, metadata_embeddings.T).diagonal()

    # Features to keep
    features_to_keep = [
        "store",
        "average_rating",
        "year",
        "month",
        "day",
        "hour",
        "rating",
        "helpful_vote",
        "rating_number",
        "num_reviews",
        "rating_deviation",
        "price",
        "similarity_good",
        "similarity_bad",
        "similarity_expensive",
        "similarity_scam",
        "similarity_error",
        "length_char",
        "length_word",
        "interaction_score",
        "sentiment_score",
        "readability",
    ]
    # features removed :
    # - main_category : trained on a single category
    # - not generalizable: userid, asin, parent_asin
    # - empty : verified_purchase
    # - processed : timestamp,
    # - processed : text_metadata = title_metadata, features, description
    # - processed : text_review = title_review, text
    # feature to process better :
    # - details
    # - categories

    # save the enriched data for vizualization purpose
    if saved_name != "":
        df.to_parquet(saved_name)

    df = df[features_to_keep]

    return df
