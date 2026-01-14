"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from .api.routes import router
from .api.models import HealthResponse
from .api.rate_limit import RateLimitMiddleware, default_limiter
from .config import settings
from .logging_config import setup_logging
from . import __version__

app = FastAPI(
    title="EDDT - Engineering Department Digital Twin",
    description="Simulation framework for engineering teams and CAD/PDM/PLM workflows",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, limiter=default_limiter)

app.include_router(router, prefix="/api/v1", tags=["api"]) 


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "EDDT",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1",
    }


@app.get("/health", response_model=HealthResponse)
async def health_root():
    """Top-level health endpoint (mirrors /api/v1/health)."""
    return HealthResponse(status="healthy", version=__version__)


def custom_openapi():
    """Custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="EDDT API",
        version=__version__,
        description="Engineering Department Digital Twin API",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.on_event("startup")
async def _startup_logging():
    """Configure logging based on settings at startup."""
    setup_logging(level=settings.log_level, json_output=settings.log_json)


@app.on_event("shutdown")
async def _shutdown():
    """Gracefully shutdown simulation manager."""
    from .api.sim_manager import sim_manager
    await sim_manager.shutdown()

