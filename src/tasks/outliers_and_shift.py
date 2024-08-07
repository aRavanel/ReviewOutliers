import os
import pickle
from typing import Tuple
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.suod import SUOD
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
import pandas as pd
import numpy as np

# module imports
from src.config import logger


# ==========================================================================
# Utils functions and module
# ==========================================================================
from src.config import MODEL_NAME_OUTLIER, MODEL_PATH_OUTLIER
from src.utils.preprocessing.preprocessing import preprocess_data


def _save_model_outlier(model) -> None:
    """Save the model to a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "wb") as file:
            pickle.dump(model, file)
    except IOError as e:
        logger.error(f"Error saving the model: {e}")
        logger.info(f"Current Directory: {os.getcwd()}")


def _load_model_outlier():
    """Load the model from a file using pickle."""
    try:
        with open(MODEL_PATH_OUTLIER, "rb") as file:
            model = pickle.load(file)
        return model
    except IOError as e:
        logger.error(f"Error loading the model: {e}")
        raise e


def outlier_shift_computation(
    df: pd.DataFrame, training: bool = True, outlier_on_score: bool = True, contamination: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict outliers and compute scores for the test set.
    if training = True, train the model and save it

    PYOD :
    scores : the higher the more anomalous the sample is
    labels_ : 0 for inliers, 1 for outliers

    SKLEARN :
    scores :
    - Negative scores represent outliers (the lower the more abnormal),
    - positive scores represent inliers. from -0.5 to 0.5
    labels_ :-1 for outliers, 1 for inliers
    """
    logger.debug("calling outlier_prediction")

    if training:
        match MODEL_NAME_OUTLIER:

            case "isolation_forest":
                model = IForest(n_estimators=100, max_samples="auto", contamination=contamination, random_state=42)

            case "one-class-svm":
                model = OCSVM(kernel="rbf", contamination=contamination)

            case "ensemble":
                # initialized a group of outlier detectors for acceleration
                detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20), COPOD(), IForest(n_estimators=100)]

                # decide the number of parallel process, and the combination method
                model = SUOD(base_estimators=detector_list, n_jobs=2, combination="average", verbose=False)

            case _:
                raise ValueError(f"Invalid model name: {MODEL_NAME_OUTLIER}")

        model.fit(df)
        _save_model_outlier(model)

    else:
        model = _load_model_outlier()

    scores = np.array(model.decision_function(df))  # list of scores

    if outlier_on_score:
        # Calculate IQR and determine a threshold for outliers
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        threshold_lower = Q1 - 1.5 * IQR
        threshold_upper = Q3 + 1.5 * IQR
        outliers = (scores < threshold_lower) | (scores > threshold_upper)
    else:
        # fixed contamination -> too rigid
        outliers = model.predict(df)  # list of 0 (inliner) and 1 (outlier)

    return np.array(outliers), scores


# ==========================================================================
# Exported functions
# ==========================================================================
def outlier_shift_prediction(df: pd.DataFrame, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the data and performs outlier and shift detection.
    """ """
    logger.debug("calling outlier_shift_prediction")

    Args:
        df (pd.DataFrame): The input data.
        training (bool, optional): Flag indicating whether the data is in training mode.
            Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The outlier labels and shift scores for each data point.
    """
    logger.debug("Preprocessing the data")

    # clean, enrich, encode the data.
    df = preprocess_data(df, training=False)

    # Perform outlier and shift detection
    outliers, scores = outlier_shift_computation(df, training)

    return outliers, scores


def outlier_prediction(df: pd.DataFrame, training: bool = True) -> np.ndarray:
    """
    Detects outliers in the data using the outlier detection model.

    Args:
        df (pd.DataFrame): The input data.
        training (bool, optional): Flag indicating whether the data is in training mode.
            Defaults to True.

    Returns:
        np.ndarray: The outlier labels for each data point.
    """

    logger.debug("calling outlier_prediction")

    # Clean, enrich, encode the data
    df = preprocess_data(df, training=False)

    # Perform outlier detection
    outliers, _ = outlier_shift_computation(df, training)

    return outliers


def shift_detection(df: pd.DataFrame, training: bool = True) -> np.ndarray:
    """
    Detects shifts in the data using the outlier detection model.

    Args:
        df (pd.DataFrame): The input data.
        training (bool, optional): Flag indicating whether the data is in training mode. Defaults to True.

    Returns:
        np.ndarray: The shift scores for each data point.
    """
    logger.debug("calling shift_detection")

    # Clean, enrich, encode the data.
    df = preprocess_data(df, training=False)

    # Do the prediction
    _, scores = outlier_shift_computation(df, training)

    return scores
