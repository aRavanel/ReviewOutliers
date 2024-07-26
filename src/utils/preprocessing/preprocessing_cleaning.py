import pandas as pd


# ==========================================================================
# Utils functions
# ==========================================================================


# ==========================================================================
# Exported functions
# ==========================================================================


def clean_enrich_reviews(df_review_in: pd.DataFrame) -> pd.DataFrame:
    """ """

    df_review = df_review_in.copy()

    # Expand timestamp
    df_review["timestamp"] = pd.to_datetime(df_review["timestamp"], unit="ms")  # Convert Unix timestamp to datetime
    df_review["year"] = df_review["timestamp"].dt.year
    df_review["month"] = df_review["timestamp"].dt.month
    df_review["day"] = df_review["timestamp"].dt.day
    df_review["hour"] = df_review["timestamp"].dt.hour

    # Handle missing values (ugly, to fix later)
    # -> if str fill with empty string
    # -> if numeric or categorical: fill with a dummy value
    df_review["year"] = df_review["year"].fillna(-1).astype("int")
    df_review["month"] = df_review["month"].fillna(-1).astype("int")
    df_review["day"] = df_review["day"].fillna(-1).astype("int")
    df_review["hour"] = df_review["hour"].fillna(-1).astype("int")
    df_review["rating"] = df_review["rating"].fillna(-1).astype("int")
    df_review["title"] = df_review["title"].fillna("").astype("str")
    df_review["text"] = df_review["text"].fillna("").astype("str")
    df_review["user_id"] = df_review["user_id"].fillna("").astype("str")
    df_review["helpful_vote"] = df_review["helpful_vote"].fillna(-1).astype("int")
    df_review["verified_purchase"] = df_review["verified_purchase"].fillna(-1).astype("int")

    # concatenate some text together
    # -> this way less features because each embeddings is of high dimensionality
    df_review["review_text"] = df_review["title"] + "/n/n" + df_review["text"]

    # drop some features:
    features_drop = ["timestamp", "text", "title"]  # transformed
    features_drop += ["images", "user_id"]  # too complex or not informative
    df_review = df_review.drop(columns=features_drop)

    # Drop some elements that cannot be used
    df_review.drop_duplicates(inplace=True)
    df_review = df_review.dropna(subset=["asin", "parent_asin"])  # drop if 'asin' or 'parent_asin'  not filled

    return df_review


def clean_enrich_metadata(df_metadata_in: pd.DataFrame) -> pd.DataFrame:
    """ """
    df_metadata = df_metadata_in.copy()

    # Handle missing values
    # features -> concatenated in a string
    df_metadata["main_category"] = df_metadata["main_category"].fillna("").astype("category")
    df_metadata["title"] = df_metadata["title"].fillna("").astype("str")
    df_metadata["average_rating"] = df_metadata["average_rating"].fillna(-1).astype("float")
    df_metadata["rating_number"] = df_metadata["rating_number"].fillna(-1).astype("int")
    df_metadata["features"] = (
        df_metadata["features"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    )
    df_metadata["store"] = df_metadata["store"].fillna("").astype("category")
    df_metadata["parent_asin"] = df_metadata["parent_asin"].fillna("").astype("str")
    # df_metadata['description'] = df_metadata['description'].fillna('')
    # df_metadata['price'] = df_metadata['price'].fillna(0)

    # concatenate some text together
    # -> this way less features because each embeddings is of high dimensionality
    df_metadata["metadata_text"] = df_metadata[["title", "features"]].astype(str).agg("/n/n".join, axis=1)

    # Drop some features
    # - `images` (list of str): too complex
    # - `videos` (list of str): too complex
    # - `details` (dict): mostly empty
    # - `categories`: mostly empty
    # - `bought_together` (boolean): mostly empty
    # - `price` (float): mostly empty
    # - `description` (list of str): mostly empty
    features_to_drop = ["title", "features"]
    features_to_drop += ["images", "videos", "details", "categories", "bought_together", "price", "description"]
    df_metadata = df_metadata.drop(columns=features_to_drop)

    # Drop samples
    df_metadata.drop_duplicates(inplace=True)
    df_metadata = df_metadata.dropna(subset=["parent_asin"])  # Drop rows if 'asin' or 'parent_asin' are not filled

    return df_metadata
