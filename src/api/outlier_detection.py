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
def detect_outliers(request: OutlierDetectionRequest):

    # format the data
    features = request.features
    text = request.text
    df = pd.DataFrame([{"text": text, "features": features}])

    # run the outlier algorithm
    list_outlier, list_score = outlier_detection(df, training=False)

    # format the output data to have valid output format
    results = [
        OutlierDetectionResponse(is_outlier=is_outlier, score=score) for is_outlier, score in zip(list_outlier, list_score)
    ]
    return BatchOutlierDetectionResponse(results=results)
