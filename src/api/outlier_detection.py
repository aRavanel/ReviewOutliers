from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from src.tasks.outliers_and_shift import outlier_prediction
from src.config import logger
from src.api.utils import BatchDetectionRequest, convert_requests_to_dataframe

# ==========================================================================
# module variables and classes
# ==========================================================================
router = APIRouter()


# Pydantic Schema for Outliers
class OutlierDetectionResponse(BaseModel):
    is_outlier: int


class BatchOutlierDetectionResponse(BaseModel):
    results: List[OutlierDetectionResponse]


# ==========================================================================
# Exported functions
# ==========================================================================


@router.post("/detect_outliers", response_model=BatchOutlierDetectionResponse)
def detect_outliers(request: BatchDetectionRequest) -> BatchOutlierDetectionResponse:
    """
    Detect outliers in a batch of data.

    Args:
        request (BatchDetectionRequest): The input data in the form of a batch request.

    Returns:
        BatchOutlierDetectionResponse: The outlier labels for each data point in the batch.
    """
    # Log the API call
    logger.debug("calling route detect_outliers")

    # Convert the requests to a dataframe
    df = convert_requests_to_dataframe(request.requests)

    # Log the dataframe used
    logger.debug("df used :")
    logger.debug(df)

    # Perform outlier detection
    list_outlier = outlier_prediction(df, training=False)

    # Create a list of OutlierDetectionResponse objects
    results = [OutlierDetectionResponse(is_outlier=outlier) for outlier in list_outlier]

    # Return the response
    return BatchOutlierDetectionResponse(results=results)
