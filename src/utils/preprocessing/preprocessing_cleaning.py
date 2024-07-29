import pandas as pd


# ==========================================================================
# Utils functions
# ==========================================================================


def clean_enrich(df_in: pd.DataFrame) -> pd.DataFrame:
    """ """
    df = df_in.copy()

    # Expand timestamp
    df["timestamp_review"] = pd.to_datetime(df["timestamp_review"], unit="ms")
    df["year_review"] = df["timestamp_review"].dt.year
    df["month_review"] = df["timestamp_review"].dt.month
    df["day_review"] = df["timestamp_review"].dt.day
    df["hour_review"] = df["timestamp_review"].dt.hour

    # Handle missing values - categories
    df["main_category_metadata"] = df["main_category_metadata"].fillna("unknown").astype("category")
    df["store_metadata"] = df["store_metadata"].fillna("unknown").astype("category")

    # Handle missing values - str
    df["title_review"] = df["title_review"].fillna("").astype("str")
    df["title_metadata"] = df["title_metadata"].fillna("").astype("str")
    df["text_review"] = df["text_review"].fillna("").astype("str")
    df["user_id_review"] = df["user_id_review"].fillna("").astype("str")
    df["features_metadata"] = (
        df["features_metadata"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    )
    df["metadata_text"] = df[["title_metadata", "features_metadata"]].astype(str).agg("/n/n".join, axis=1)
    # df_metadata['description'] = df_metadata['description'].fillna('').astype("str")

    # Handle missing values - numeric
    numeric_fields = {
        "year_review",
        "month_review",
        "day_review",
        "hour_review",
        "rating_review",
        "helpful_vote_review",
        "verified_purchase_review",
        "rating_number_metadata",
    }
    for field in numeric_fields:
        df[field] = df[field].fillna(-1).astype(int)
    df["average_rating_metadata"] = df["average_rating_metadata"].fillna(-1).astype("float")
    # df_metadata['price'] = df_metadata['price'].fillna(-1).astype("float")

    # concatenate some text together
    # -> this way less features because each embeddings is of high dimensionality
    df["review_text"] = df["title"] + "/n/n" + df["text"]
    df["metadata_text"] = df[["title", "features"]].astype(str).agg("/n/n".join, axis=1)

    # Drop features that are too complex, not informative, or have been transformed
    features_to_drop = [
        "timestamp_review",  # processed
        "title_review",  # processed
        "text_review",  # processed
        "title_metadata",  # processed
        "features_metadata",  # processed
        "user_id_review",  #
        "images_review",  # too complex
        "images_metadata",  # too complex
        "videos_metadata",  # too complex
        "details_metadata",  # mostly empty
        "categories_metadata",  # mostly empty
        "bought_together_metadata",  # mostly empty
        "price_metadata",  # mostly empty
        "description_metadata",  # mostly empty
    ]
    df.drop(columns=features_to_drop, inplace=True)

    # Drop duplicates and rows with missing 'asin' or 'parent_asin'
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["asin", "parent_asin"])

    return df


# ==========================================================================
# Exported functions
# ==========================================================================


def clean_data(df_review, df_metadata) -> pd.DataFrame:
    """ """
    # merge data
    df_merged = pd.merge(df_review, df_metadata, on="parent_asin", how="inner", suffixes=("_review", "_metadata"))

    # clean it
    df_merged = clean_enrich(df_merged)

    return df_merged
