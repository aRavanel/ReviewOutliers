from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from typing import List
import os
import pandas as pd

# module imports
from src.utils.preprocessing.preprocessing import preprocess_data
from src.tasks.distribution_shift import distribution_shift_scoring
from logger_config import logger


# ==========================================================================
# Schema and module variables
# ==========================================================================


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
    timestamp: datetime
    helpful_vote: int
    verified_purchase: bool


class BatchDistributionShiftRequest(BaseModel):
    requests: List[DistributionShiftRequest]


class DistributionShiftResponse(BaseModel):
    shift_score: float


class BatchDistributionShiftResponse(BaseModel):
    responses: List[DistributionShiftResponse]


router = APIRouter()

# load the model

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
