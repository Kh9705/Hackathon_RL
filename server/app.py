from openenv.core.env_server import create_fastapi_app
from .environment import SCEnv
from envs.sc_env.models import SCAct, SCObs

# Pass the CLASS SCEnv, not the instance SCEnv()
app = create_fastapi_app(SCEnv, action_cls=SCAct, observation_cls=SCObs)