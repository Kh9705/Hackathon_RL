"""
FastAPI Server for Supply Chain Environment.

This module creates the OpenEnv-compliant REST API for the supply chain
simulation environment. It exposes endpoints for interaction with the
environment following OpenEnv framework standards.

Key Responsibilities:
- Create and configure FastAPI application
- Register environment with OpenEnv framework
- Generate required REST endpoints (reset, step, state, health)
- Handle environment lifecycle (creation, reset, step execution)

OpenEnv Endpoints Generated:
- POST /reset          : Reset environment, return initial observation
- POST /step           : Execute action, return (obs, reward, done)
- GET /state           : Get internal state (for debugging)
- GET /health          : Health check endpoint

Example:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from typing import Optional, Dict, Any

# ============================================================================
# OPENENV FRAMEWORK IMPORTS
# ============================================================================

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    # Fallback for different framework versions
    from openenv.core import create_fastapi_app

# ============================================================================
# PROJECT-SPECIFIC IMPORTS
# ============================================================================
# Using relative imports within server package

from .environment import SCEnv      # Environment class
from .model import SCAct, SCObs      # Action and Observation models


def create_app():
    """
    Create and configure the FastAPI application with OpenEnv integration.
    
    The OpenEnv framework's create_fastapi_app() automatically generates:
    - HTTP endpoints for reset/step/state/health
    - Request/response serialization using Pydantic models
    - Environment instance management
    
    CRITICAL: Pass environment CLASS (callable), not an instance!
    
    Returns:
        FastAPI: Configured FastAPI application ready to serve
        
    Example:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    
    # =====================================================================
    # OPENENV INTEGRATION
    # =====================================================================
    # create_fastapi_app handles:
    # 1. Environment instantiation per request (stateless HTTP)
    # 2. Action deserialization (JSON → SCAct)
    # 3. Observation serialization (SCObs → JSON)
    # 4. Error handling and response formatting
    
    app = create_fastapi_app(
        SCEnv,                    # Environment CLASS (not instance)
        action_cls=SCAct,         # Action model for request body
        observation_cls=SCObs     # Observation model for response
    )
    
    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================
# Create the app at module level (required by uvicorn server)

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