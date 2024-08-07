from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Any, Dict
from datetime import datetime
import pandas as pd

# module imports
from src.tasks.outliers_and_shift import outlier_prediction
from src.config import logger
import logging

router = APIRouter()

# ==========================================================================
# Pydantic Schema
# ==========================================================================


# Define request and response models
class OutlierDetectionRequest(BaseModel):
    main_category: str
    title_review: str
    average_rating: float
    rating_number: int
    features: List[str]
    store: str
    rating: float
    title_metadata: str
    text: str
    timestamp: str
    helpful_vote: int
    verified_purchase: bool


class BatchOutlierDetectionRequest(BaseModel):
    requests: List[OutlierDetectionRequest]


class OutlierDetectionResponse(BaseModel):
    is_outlier: int
    score: float


class BatchOutlierDetectionResponse(BaseModel):
    results: List[OutlierDetectionResponse]


# ==========================================================================
# Exported Utilities
# ==========================================================================


def create_outlier_request(data: Dict[str, Any]) -> OutlierDetectionRequest:
    """
    To Force the data into the shema. Useful for debugging.
    """
    logger.debug("calling create_outlier_request")
    logger.debug("compute dummy_values")

    dummy_values = {
        "main_category": "unknown",
        "title_review": "No title",
        "average_rating": 0.0,
        "rating_number": 0,
        "features": [],
        "store": "unknown",
        "rating": 0.0,
        "title_metadata": "No title",
        "text": "No text",
        "timestamp": datetime(1970, 1, 1).isoformat(),
        "helpful_vote": 0,
        "verified_purchase": False,
    }
    logger.debug(dummy_values)

    filled_data = {}
    for field in OutlierDetectionRequest.model_fields:
        logger.debug(field)
        if field in data:
            if field == "timestamp" and isinstance(data[field], (pd.Timestamp, datetime)):
                filled_data[field] = data[field].isoformat()
            else:
                filled_data[field] = data[field]
            logger.debug(f"{field} ok")

        else:
            filled_data[field] = dummy_values[field]
            logging.warning(f"Missing field '{field}', filled with dummy value '{dummy_values[field]}'.")

    return OutlierDetectionRequest(**filled_data)


def create_batch_outlier_request(data_list: List[Dict[str, Any]]) -> BatchOutlierDetectionRequest:
    logger.debug("calling create_batch_outlier_request")
    requests = [create_outlier_request(data) for data in data_list]
    return BatchOutlierDetectionRequest(requests=requests)


# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/detect_outliers", response_model=BatchOutlierDetectionResponse)
def detect_outliers(request: BatchOutlierDetectionRequest) -> BatchOutlierDetectionResponse:

    logger.debug("calling route detect_outliers")

    data = []
    for req in request.requests:
        data_i = {
            "main_category": req.main_category,
            "title_review": req.title_review,
            "average_rating": req.average_rating,
            "rating_number": req.rating_number,
            "features": req.features,
            "store": req.store,
            "rating": req.rating,
            "title_metadata": req.title_metadata,
            "text": req.text,
            "timestamp": req.timestamp,
            "helpful_vote": req.helpful_vote,
            "verified_purchase": req.verified_purchase,
        }
        logger.debug(f"data_i : {data_i}")

        data.append(data_i)
    df = pd.DataFrame([data])

    logger.debug("df used :")
    logger.debug(df)

    # run the outlier algorithm
    list_outlier, list_score = outlier_prediction(df, training=False)

    # format the output data to have valid output format
    results = [
        OutlierDetectionResponse(is_outlier=is_outlier, score=score) for is_outlier, score in zip(list_outlier, list_score)
    ]
    return BatchOutlierDetectionResponse(results=results)
