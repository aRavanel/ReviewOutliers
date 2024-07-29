from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

# module imports
from src.tasks.outliers import outliers_inference


# ==========================================================================
# Shema and module variables
# ==========================================================================


# Define request and response models
class OutlierDetectionRequest(BaseModel):
    text: str
    features: List[float]


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
    list_outlier, list_score = outliers_inference(df)

    # format the output data to have valid output format
    results = [
        OutlierDetectionResponse(is_outlier=is_outlier, score=score) for is_outlier, score in zip(list_outlier, list_score)
    ]
    return BatchOutlierDetectionResponse(results=results)
