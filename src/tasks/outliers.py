import os
import pickle
from typing import Tuple
from pyod.models.iforest import IForest
import pandas as pd
import numpy as np

# module imports
from src.config import logger


# ==========================================================================
# Utils functions and module
# ==========================================================================
from src.config import MODEL_NAME_OUTLIER, MODEL_PATH_OUTLIER
from src.utils.preprocessing.preprocessing import preprocess_data


def save_model_outlier(model) -> None:
    """Save the model to a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        logger.error(f"Error saving the model: {e}")
        logger.info(f"Current Directory: {os.getcwd()}")


def load_model_outlier():
    """Load the model from a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "rb") as file:
            model = pickle.load(file)
        return model
    except IOError as e:
        logger.error(f"Error loading the model: {e}")
        raise e


# ==========================================================================
# Exported functions
# ==========================================================================


def outlier_detection(df: pd.DataFrame, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the Isolation Forest model on the training set and optionally save it.
    scores > 0   ->   outlier
    Predict outliers and compute scores for the test set.
    """
    logger.debug("calling outlier_detection")

    logger.debug("DataFrame being passed to the model:")
    logger.debug(df)

    # clean, enrich, encode the data
    # TODO : have one specific preprocessing per model
    df = preprocess_data(df, training=False)

    # outlier detection
    match MODEL_NAME_OUTLIER:
        case "isolation_forest":
            if training:
                # 0.1 -> expect 10% of outliers
                model = IForest(n_estimators=100, max_samples="auto", contamination=0.1, random_state=42)
                model.fit(df)
                save_model_outlier(model)
            else:
                model = load_model_outlier()

            # run the prediction - v0 (fixed contamination -> too rigid)
            # outliers = model.predict(df)  # list of 0 (inliner) and 1 (outlier)
            # scores = model.decision_function(df)  # list of scores

            # run the prediction - v1 : Calculate IQR and determine a threshold for outliers
            scores = np.array(model.decision_function(df))  # list of scores
            Q1 = np.percentile(scores, 25)
            Q3 = np.percentile(scores, 75)
            IQR = Q3 - Q1
            threshold_lower = Q1 - 1.5 * IQR
            threshold_upper = Q3 + 1.5 * IQR
            outliers = (scores < threshold_lower) | (scores > threshold_upper)

        case _:
            raise ValueError(f"Invalid model name: {MODEL_NAME_OUTLIER}")

    return np.array(outliers), np.array(scores)
