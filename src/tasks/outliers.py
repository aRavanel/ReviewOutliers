import pickle
from typing import List
from pyod.models.iforest import IForest
import pandas as pd

# module imports
import src.config as CONFIG

# ==========================================================================
# Utils functions and module
# ==========================================================================


def save_model(model, filepath: str):
    """Save the model to a file using pickle."""
    with open(filepath, "wb") as file:
        pickle.dump(model, file)


def load_model(filepath: str):
    """Load the model from a file using pickle."""
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model


# ==========================================================================
# Exported functions
# ==========================================================================


def outliers_train(train_df: pd.DataFrame, save_path: str | None = None):
    """
    Train the Isolation Forest model on the training set and optionally save it.
    """

    # create the model
    if CONFIG.MODEL_NAME == "isolation_forest":
        model = IForest(contamination=0.1, random_state=42)  # 0.1 -> expect 10% of outliers
    else:
        raise ValueError(f"Invalid model name: {CONFIG.MODEL_NAME}")

    # train the model
    model.fit(train_df)

    if save_path:
        save_model(model, save_path)

    return model


def outliers_inference(df: pd.DataFrame) -> tuple[List[int], List[float]]:
    """
    scores > 0   ->   outlier
    Predict outliers and compute scores for the test set.
    """

    # load the model
    model = load_model(CONFIG.FILENAME_OUTLIER)

    # run the prediction
    outliers_test = model.predict(df)  # list of 0 (inliner) and 1 (outlier)
    scores_test = model.decision_function(df)  # list of scores

    return outliers_test, scores_test
