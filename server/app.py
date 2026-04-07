"""
FastAPI server for Supply Chain Environment.

This module creates the OpenEnv-compliant API endpoints.
Adjust the import paths based on your project structure.
"""

import uvicorn
from typing import Optional, Dict, Any

# Import framework components
try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    # Fallback for different versions
    from openenv.core import create_fastapi_app

# Import environment and models
# Adjust paths based on your directory structure
from server.environment import SCEnv
from envs.sc_env.models import SCAct, SCObs


def create_app():
    """
    Create and configure the FastAPI application.
    
    This uses the OpenEnv framework's create_fastapi_app helper
    which automatically generates the required endpoints:
    - POST /reset
    - POST /step
    - GET /state
    - GET /health
    """
    
    # create_fastapi_app expects the environment CLASS (callable), not instance
    app = create_fastapi_app(
        SCEnv,
        action_cls=SCAct,
        observation_cls=SCObs
    )
    
    return app


# Create the app instance (required by uvicorn)
app = create_app()


def main():
    """
    Entry point for running the server.
    Called by the [project.scripts] in pyproject.toml.
    
    Usage:
        python -m server.app
        # OR
        uvicorn server.app:app --host 0.0.0.0 --port 8000
    """
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Don't reload in production
    )


if __name__ == "__main__":
    main()