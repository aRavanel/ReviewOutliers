import pandas as pd

# module imports
from logger_config import logger

# ==========================================================================
# Utils functions
# ==========================================================================


# ==========================================================================
# Exported functions
# ==========================================================================


def clean_enrich(df_in: pd.DataFrame) -> pd.DataFrame:
    """ """
    df = df_in.copy()

    # Expand timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    # Handle missing values - categories
    df["main_category"] = df["main_category"].fillna("unknown").astype("category")
    df["store"] = df["store"].fillna("unknown").astype("category")

    # Handle missing values - str
    # concatenate some text together -> this way less features because each embeddings is of high dimensionality
    # df_metadata['description'] = df_metadata['description'].fillna('').astype("str")
    df["user_id"] = df["user_id"].fillna("").astype("str")

    df["title_review"] = df["title_review"].fillna("").astype("str")
    df["text"] = df["text"].fillna("").astype("str")
    df["text_review"] = df["title_review"] + "/n/n" + df["text"]

    df["title_metadata"] = df["title_metadata"].fillna("").astype("str")
    df["features"] = df["features"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    df["text_metadata"] = df[["title_metadata", "features"]].astype(str).agg("/n/n".join, axis=1)

    # Handle missing values - numeric
    numeric_fields = {
        "year",
        "month",
        "day",
        "hour",
        "rating",
        "helpful_vote",
        "verified_purchase",
        "rating_number",
    }
    for field in numeric_fields:
        df[field] = df[field].fillna(-1).astype(int)
    df["average_rating"] = df["average_rating"].fillna(-1).astype("float")
    # df_metadata['price'] = df_metadata['price'].fillna(-1).astype("float")

    # Drop features that are too complex, not informative, or have been transformed
    features_to_drop = [
        "timestamp",  # processed
        "title_review",  # processed
        "title_metadata",  # processed
        "features",  # processed
        "user_id",  #
        "images_review",  # too complex
        "images_metadata",  # too complex
        "videos",  # too complex
        "details",  # mostly empty
        "categories",  # mostly empty
        "bought_together",  # mostly empty
        "price",  # mostly empty
        "description",  # mostly empty
    ]
    df.drop(columns=features_to_drop, inplace=True, errors="ignore")  # errors='ignore' in case already dropped
    df.drop_duplicates(inplace=True)

    return df
