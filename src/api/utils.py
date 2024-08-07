from pydantic import BaseModel
from typing import List, Any, Dict, Optional
from datetime import datetime
import pandas as pd
from src.config import logger


# Common Pydantic Schema
class DetectionRequest(BaseModel):
    user_id: Optional[str] = None
    title_review: Optional[str] = "No title"
    text: Optional[str] = "No text"
    title_metadata: Optional[str] = "No title"
    features: Optional[str] = ""
    description: Optional[str] = ""
    main_category: Optional[str] = "unknown"
    store: Optional[str] = "unknown"
    asin: Optional[str] = None
    parent_asin: Optional[str] = None
    timestamp: Optional[str] = datetime(1970, 1, 1).isoformat()
    rating: Optional[int] = 0
    average_rating: Optional[float] = 0.0
    price: Optional[float] = -1.0
    helpful_vote: Optional[int] = 0
    verified_purchase: Optional[bool] = False
    rating_number: Optional[int] = 0


class BatchDetectionRequest(BaseModel):
    requests: List[DetectionRequest]


def create_detection_request(data: Dict[str, Any]) -> DetectionRequest:
    """
    Force the data into the schema. Useful for debugging.
    """
    logger.debug("calling create_detection_request")
    filled_data = {}
    for field, value in DetectionRequest.__fields__.items():
        if field in data:
            if field == "timestamp" and isinstance(data[field], (pd.Timestamp, datetime)):
                filled_data[field] = data[field].isoformat()
            else:
                filled_data[field] = data[field]
            logger.debug(f"{field} found in input data")
        else:
            filled_data[field] = value.default
            logger.warning(f"Missing field '{field}', filled with default value '{value.default}'.")

    return DetectionRequest(**filled_data)


def create_batch_detection_request(data_list: List[Dict[str, Any]]) -> BatchDetectionRequest:
    """
    Create a BatchDetectionRequest object from a list of dictionaries.

    Args:
        data_list (List[Dict[str, Any]]): A list of dictionaries containing DetectionRequest data.

    Returns:
        BatchDetectionRequest: A BatchDetectionRequest object containing the DetectionRequests.
    """
    # Log the API call
    logger.debug("calling create_batch_detection_request")

    # Create a list of DetectionRequest objects from the data
    requests = [create_detection_request(data) for data in data_list]

    # Create a BatchDetectionRequest object with the requests
    return BatchDetectionRequest(requests=requests)


def convert_requests_to_dataframe(requests: List[DetectionRequest]) -> pd.DataFrame:
    """
    Convert a list of DetectionRequest objects to a pandas DataFrame.
    """
    data = [{field: getattr(req, field) for field in req.__fields__} for req in requests]
    df = pd.DataFrame(data)
    logger.debug("Converted requests to DataFrame")
    logger.debug(df)
    return df
