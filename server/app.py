import uvicorn
from openenv.core.env_server import create_fastapi_app
from .environment import SCEnv
from envs.sc_env.models import SCAct, SCObs

# 1. Create the app object
app = create_fastapi_app(SCEnv, action_cls=SCAct, observation_cls=SCObs)

# 2. Add the main() function (REQUIRED by the grader)
def main():
    # Use the string "server.app:app" so uvicorn finds the file correctly
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

# 3. Add the execution block (REQUIRED by the grader)
if __name__ == "__main__":
    main()