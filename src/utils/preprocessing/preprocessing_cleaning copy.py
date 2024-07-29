import pandas as pd


# ==========================================================================
# Utils functions
# ==========================================================================
def clean_enrich_reviews(df_review_in: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the reviews DataFrame.

    Parameters:
    - df_review_in (pd.DataFrame): Input DataFrame containing review data.

    Returns:
    - pd.DataFrame: Cleaned and enriched DataFrame.
    """

    df = df_review_in.copy()

    # Expand timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert Unix timestamp to datetime
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    # Handle missing values (ugly, to fix later)
    # -> if str fill with empty string
    # -> if numeric or categorical: fill with a dummy value
    df["year"] = df["year"].fillna(-1).astype("int")
    df["month"] = df["month"].fillna(-1).astype("int")
    df["day"] = df["day"].fillna(-1).astype("int")
    df["hour"] = df["hour"].fillna(-1).astype("int")
    df["rating"] = df["rating"].fillna(-1).astype("int")
    df["title"] = df["title"].fillna("").astype("str")
    df["text"] = df["text"].fillna("").astype("str")
    df["user_id"] = df["user_id"].fillna("").astype("str")
    df["helpful_vote"] = df["helpful_vote"].fillna(-1).astype("int")
    df["verified_purchase"] = df["verified_purchase"].fillna(-1).astype("int")

    # Concatenate title and text to form review_text
    # -> this way less features because each embeddings is of high dimensionality
    df["review_text"] = df["title"] + "/n/n" + df["text"]

    # drop some features:
    features_drop = ["timestamp", "text", "title"]  # transformed
    features_drop += ["images", "user_id"]  # too complex or not informative
    df = df.drop(columns=features_drop)

    # Drop duplicates and rows with missing 'asin' or 'parent_asin'
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["asin", "parent_asin"])

    return df


def clean_enrich_metadata(df_metadata_in: pd.DataFrame) -> pd.DataFrame:
    """ """
    df = df_metadata_in.copy()

    # Handle missing values
    # features -> concatenated in a string
    df["main_category"] = df["main_category"].fillna("").astype("category")
    df["title"] = df["title"].fillna("").astype("str")
    df["average_rating"] = df["average_rating"].fillna(-1).astype("float")
    df["rating_number"] = df["rating_number"].fillna(-1).astype("int")
    df["features"] = df["features"].apply(lambda x: " ".join(x) if isinstance(x, list) and x else "").astype("str")
    df["store"] = df["store"].fillna("").astype("category")
    df["parent_asin"] = df["parent_asin"].fillna("").astype("str")
    # df_metadata['description'] = df_metadata['description'].fillna('')
    # df_metadata['price'] = df_metadata['price'].fillna(0)

    # concatenate some text together
    # -> this way less features because each embeddings is of high dimensionality
    df["metadata_text"] = df[["title", "features"]].astype(str).agg("/n/n".join, axis=1)

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
    df = df.drop(columns=features_to_drop)

    # Drop samples
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["parent_asin"])  # Drop rows if 'asin' or 'parent_asin' are not filled

    return df


# ==========================================================================
# Exported functions
# ==========================================================================


def clean_data(df_review, df_metadata) -> pd.DataFrame:
    """ """

    # cleaning
    df_review = clean_enrich_reviews(df_review)
    df_metadata = clean_enrich_metadata(df_metadata)

    # Merge the datasets on 'parent_asin' with suffixes for duplicate columns
    merged_df = pd.merge(df_review, df_metadata, on="parent_asin", how="inner", suffixes=("_review", "_metadata"))
    merged_df = merged_df.drop(columns=["parent_asin", "asin"])  # those features will be OOD at inference for new data

    return merged_df
