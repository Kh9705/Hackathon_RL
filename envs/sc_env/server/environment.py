from openenv.core.env_server import Environment
from envs.sc_env.models import SCAct, SCObs

class SCEnv(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs):
        # 1. Handle the seed
        if seed is not None:
            pass

        # 2. Reset your internal supply chain state
        self.current_step = 0
        self.p_ord = 0     # Pending orders
        self.t_cap = 100   # Total capacity

        # 3. Return the SCObs object directly (Matching step and state)
        return SCObs(
            p_ord=self.p_ord, 
            t_cap=self.t_cap,
            reward=0.0,    # Default starting reward
            done=False,    # Obviously not done on reset
            info={}
        )

    def step(self, a: SCAct):
        # Return ONLY the observation object. The server pulls reward/done from it.
        # (In a real scenario, you'd calculate this based on the action 'a')
        obs = SCObs(t_cap=100, p_ord=20, reward=1.0, done=False, info={})
        return obs

    def state(self) -> SCObs:
        return SCObs(t_cap=100, p_ord=50, reward=0.0, done=False, info={})