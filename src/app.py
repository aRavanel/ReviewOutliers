from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# module imports
from src.api import outlier_detection, distribution_shift
from src.config import logger

# ==========================================================================
# Module variables
# ==========================================================================

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
app.include_router(outlier_detection.router, prefix="/api/detect_outliers")
app.include_router(distribution_shift.router, prefix="/api/distribution_shift")

# ==========================================================================
# util functions
# ==========================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch and log all exceptions.
    """
    logger.error(f"An error occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )


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
