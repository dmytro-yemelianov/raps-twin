"""Entry point for running EDDT as a module or console script."""

import uvicorn
from .config import settings


def main():
    """Run the API server."""
    uvicorn.run(
        "eddt.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
