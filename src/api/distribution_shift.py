from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List


# ==========================================================================
# Shema and module variables
# ==========================================================================


# Define request and response models
class DistributionShiftRequest(BaseModel):
    text: str
    features: List[float]


class DistributionShiftResponse(BaseModel):
    shift_score: float


router = APIRouter()

# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/distribution_shift", response_model=DistributionShiftResponse)
def distribution_shift(request: DistributionShiftRequest):
    # Placeholder for actual distribution shift scoring logic
    shift_score = 0.0

    # Implement distribution shift scoring here
    # shift_score = distribution_shift_function(request.text, request.features)

    return DistributionShiftResponse(shift_score=shift_score)
