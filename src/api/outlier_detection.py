from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List


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
    is_outlier: bool
    score: float


class BatchOutlierDetectionResponse(BaseModel):
    requests: List[OutlierDetectionResponse]


router = APIRouter()

# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/detect_outliers", response_model=OutlierDetectionResponse)
def detect_outliers(request: OutlierDetectionRequest):
    # Placeholder for actual outlier detection logic
    is_outlier = False
    score = 0.0

    # Implement outlier detection here
    # is_outlier, score = outlier_detection_function(request.text, request.features)

    return OutlierDetectionResponse(is_outlier=is_outlier, score=score)
