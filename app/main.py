from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from utils.config import settings
from utils.logging import logger


def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    application = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        # Add version, docs_url, etc. here if needed
    )

    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    application.include_router(api_router)

    # Add startup event
    @application.on_event("startup")
    async def startup_event():
        logger.info("Starting application...")

    # Add shutdown event
    @application.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down application...")

    return application


# Create the app instance
app = create_application()


if __name__ == "__main__":
    # This block is used when running directly with Python
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)