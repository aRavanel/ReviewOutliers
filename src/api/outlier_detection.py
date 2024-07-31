from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime
import pandas as pd

# module imports
from src.tasks.outliers import outlier_detection


# ==========================================================================
# Shema and module variables
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
    timestamp: datetime
    helpful_vote: int
    verified_purchase: bool


class BatchOutlierDetectionRequest(BaseModel):
    requests: List[OutlierDetectionRequest]


class OutlierDetectionResponse(BaseModel):
    is_outlier: int
    score: float


class BatchOutlierDetectionResponse(BaseModel):
    results: List[OutlierDetectionResponse]


router = APIRouter()

# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/detect_outliers", response_model=OutlierDetectionResponse)
def detect_outliers(request: BatchOutlierDetectionRequest):

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

    # run the outlier algorithm
    list_outlier, list_score = outlier_detection(df, training=False)

    # format the output data to have valid output format
    results = [
        OutlierDetectionResponse(is_outlier=is_outlier, score=score) for is_outlier, score in zip(list_outlier, list_score)
    ]
    return BatchOutlierDetectionResponse(results=results)
