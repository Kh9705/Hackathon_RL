from openenv.core.env_server import Environment
from envs.sc_env.models import SCAct, SCObs

class SCEnv(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs):
        # 1. Handle the seed (optional but good practice)
        if seed is not None:
            # Set your random seeds here if needed
            pass

        # 2. Reset your internal supply chain state
        self.current_step = 0
        self.p_ord = 0     # Pending orders
        self.t_cap = 100   # Total capacity (example)

        # 3. Return the EXACT format your SCObs model expects
        # (Make sure these keys perfectly match your models.py)
        observation = {
            "p_ord": self.p_ord,
            "t_cap": self.t_cap
        }
        
        # OpenEnv usually expects just the observation object/dict to be returned
        return observation

    def step(self, a: SCAct):
        # Return ONLY the observation object. The server pulls reward/done from it.
        obs = SCObs(t_cap=100, p_ord=20, reward=1.0, done=False, info={})
        return obs

    def state(self) -> SCObs:
        return SCObs(t_cap=100, p_ord=50, reward=0.0, done=False, info={})