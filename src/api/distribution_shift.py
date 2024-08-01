from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from typing import List, Any, Dict
import os
import pandas as pd

# module imports
from src.utils.preprocessing.preprocessing import preprocess_data
from src.tasks.distribution_shift import distribution_shift_scoring
from src.config import logger
import logging

# ==========================================================================
# Schema and module variables
# ==========================================================================
router = APIRouter()


# Define request and response models
class DistributionShiftRequest(BaseModel):
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


class BatchDistributionShiftRequest(BaseModel):
    requests: List[DistributionShiftRequest]


class DistributionShiftResponse(BaseModel):
    shift_score: float


class BatchDistributionShiftResponse(BaseModel):
    responses: List[DistributionShiftResponse]


# ==========================================================================
# Utilities
# ==========================================================================


def create_distribution_shift_request(data: Dict[str, Any]) -> DistributionShiftRequest:
    """
    To Force the data into the shema. Useful for debugging.
    """
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
        "timestamp": datetime(1970, 1, 1),
        "helpful_vote": 0,
        "verified_purchase": False,
    }

    filled_data = {}
    for field in DistributionShiftRequest.model_fields:
        if field in data:
            if field == "timestamp" and isinstance(data[field], (pd.Timestamp, datetime)):
                filled_data[field] = data[field].isoformat()
            else:
                filled_data[field] = data[field]
        else:
            filled_data[field] = dummy_values[field]
            logging.warning(f"Missing field '{field}', filled with dummy value '{dummy_values[field]}'.")
    return DistributionShiftRequest(**filled_data)


def create_batch_distribution_shift_request(data_list: List[Dict[str, Any]]) -> BatchDistributionShiftRequest:
    requests = [create_distribution_shift_request(data) for data in data_list]
    return BatchDistributionShiftRequest(requests=requests)


# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/distribution_shift", response_model=BatchDistributionShiftResponse)
def distribution_shift(request: BatchDistributionShiftRequest):
    logger.debug("calling distribution_shift")

    # Construct a DataFrame from the list of requests
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
        data.append(data_i)
    df = pd.DataFrame([data])

    # Preprocess the data
    df = preprocess_data(df, training=False)

    # Apply the distribution shift function to each processed feature set
    df["shift_score"] = df["processed_features"].apply(distribution_shift_scoring)

    # Convert the results to a list of DistributionShiftResponse
    responses = [DistributionShiftResponse(shift_score=row["shift_score"]) for index, row in df.iterrows()]

    return BatchDistributionShiftResponse(responses=responses)
