from openenv.core.env_server import Environment
from envs.sc_env.models import SCAct, SCObs

class SCEnv(Environment):
    def reset(self):
        return SCObs(t_cap=100, p_ord=50, reward=0.0, done=False, info={})

    def step(self, a: SCAct):
        # Return ONLY the observation object. The server pulls reward/done from it.
        obs = SCObs(t_cap=100, p_ord=20, reward=1.0, done=False, info={})
        return obs

    def state(self) -> SCObs:
        return SCObs(t_cap=100, p_ord=50, reward=0.0, done=False, info={})