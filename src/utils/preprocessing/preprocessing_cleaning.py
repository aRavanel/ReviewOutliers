import pandas as pd

# module imports
from src.config import logger

# ==========================================================================
# Utils functions
# ==========================================================================


# ==========================================================================
# Exported functions
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

    # Product based
    average_ratings = df.groupby("asin")["rating"].mean()
    df["average_rating_product"] = df["asin"].map(average_ratings)
    df["rating_deviation"] = abs(df["rating"] - df["average_rating"])

    # User based
    average_ratings_user = df.groupby("user_id")["rating"].mean()
    count_ratings_user = df.groupby("user_id")["rating"].count()
    df["average_rating_user"] = df["user_id"].map(average_ratings_user)
    df["count_rating_user"] = df["user_id"].map(count_ratings_user)

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

    # merged_df = merged_df.drop(columns=["parent_asin", "asin"])  # those features will be OOD at inference for new data

    return df
