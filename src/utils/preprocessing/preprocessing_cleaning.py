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

    # Find the most frequent label in the main_category column
    main_label = df["main_category"].value_counts().idxmax()

    # Filter the DataFrame to keep only rows with the most frequent label (some were in wrong category)
    df = df[df["main_category"] == main_label]

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

    # df_metadata['price'] = df_metadata['price'].fillna(-1).astype("float")

    # features removed :
    # - not generalizable: userid, asin, parent_asin
    # - empty : verified_purchase
    # - processed : timestamp,
    # - processed : title_metadata, features, description
    # - processed : title_review, text

    # feature to process better :
    # - details
    # - categories

    # Features to keep
    features_to_keep = [
        "main_category",
        "store",
        "text_review",
        "text_metadata",
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
    ]

    df = df[features_to_keep]

    # drop duplicates (because less columns now)
    df.drop_duplicates(inplace=True)

    return df
