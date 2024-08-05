import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade, gunning_fog, flesch_reading_ease
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# module imports
from src.config import logger

# ==========================================================================
# Module variables
# ==========================================================================
from src.config import MODEL_NAME_EMBEDDINGS

# pickled models
model_embeddings = SentenceTransformer(MODEL_NAME_EMBEDDINGS)  # text embeddings

# download resources
nltk.download("vader_lexicon")


# ==========================================================================
# Utils functions
# ==========================================================================


def clean_enrich(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    check here :
    https://github.com/yashpandey474/Identification-of-fake-reviews/blob/main/Code/Data_Processing/Feature_Extraction.py
    https://iacis.org/iis/2020/1_iis_2020_185-194.pdf
    https://www.sciencedirect.com/science/article/abs/pii/S0167923622001828

    """

    logger.debug("calling clean_enrich")

    df = df_in.copy()

    # Expand timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    # User based rating
    average_ratings_user = df.groupby("user_id")["rating"].mean()
    count_ratings_user = df.groupby("user_id")["rating"].count()
    df["average_rating_user"] = df["user_id"].map(average_ratings_user)
    df["count_rating_user"] = df["user_id"].map(count_ratings_user)
    df["rating_deviation"] = abs(df["average_rating"] - df["average_rating_user"])

    # total number of reviews
    num_reviews = df.groupby("asin").size()
    df["num_reviews"] = df["asin"].map(num_reviews)

    # Handle missing values - categories
    df["main_category"] = df["main_category"].fillna("unknown").astype("category")
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
    review_embeddings = model_embeddings.encode(df["text_review"].tolist(), show_progress_bar=True)
    metadata_embeddings = model_embeddings.encode(df["text_metadata"].tolist(), show_progress_bar=True)

    # VADER Sentiment Score
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

    # Compute readability scores
    df["flesch_kincaid"] = df["text_review"].apply(flesch_kincaid_grade)
    df["gunning_fog"] = df["text_review"].apply(gunning_fog)
    df["flesch_reading_ease"] = df["text_review"].apply(flesch_reading_ease)

    # Length-based features
    df["length_char"] = df["text_review"].apply(len)
    df["length_word"] = df["text_review"].apply(lambda x: len(x.split()))
    df["length_sentence"] = df["text_review"].apply(lambda x: len(x.split(".")))

    # Interaction features
    df["interaction_score"] = np.dot(review_embeddings, metadata_embeddings.T).diagonal()

    # Features to keep
    features_to_keep = [
        "main_category",
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
        "length_sentence",
        "interaction_score",
        "sentiment_score",
        "flesch_kincaid",
        "gunning_fog",
        "flesch_reading_ease",
    ]
    # features removed :
    # - not generalizable: userid, asin, parent_asin
    # - empty : verified_purchase
    # - processed : timestamp,
    # - processed : text_metadata = title_metadata, features, description
    # - processed : text_review = title_review, text
    # feature to process better :
    # - details
    # - categories

    df = df[features_to_keep]

    return df
