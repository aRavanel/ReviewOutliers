from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# module imports
from src.api.outlier_detection import outlier_router
from src.api.distribution_shift import shift_router

# ==========================================================================
# Module variables
# ==========================================================================
from src.config import logger

# Create a FastAPI application
app = FastAPI(
    title="Outlier Detection and Distribution Shift API",
    description="API for detecting outliers and scoring distribution shifts in data.",
    version="1.0.0",
)

# CORS support if the API will be accessed from web applications.
#  needed when API is accessed from a different domain than where frontend is hosted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust according to your security needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers for different endpoints with appropriate prefixes
app.include_router(outlier_router, prefix="/anomaly", tags=["anomaly-detection"])
app.include_router(shift_router, prefix="/anomaly", tags=["anomaly-detection"])

# ==========================================================================
# util functions
# ==========================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc.errors()}", exc_info=exc)
    return JSONResponse(status_code=400, content={"message": "Validation error", "details": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=exc)
    return JSONResponse(status_code=500, content={"message": "An internal server error occurred."})


# ==========================================================================
# Exported functions
# ==========================================================================


@app.get("/")
def read_root():
    """
    Root endpoint for the Outlier Detection and Distribution Shift API.
    Returns a welcome message.
    """
    logger.debug("calling read_root")

    return {"message": "Welcome 👋 to the Outlier Detection and Distribution Shift API"}
