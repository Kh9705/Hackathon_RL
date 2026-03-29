from typing import Dict, Any
from openenv.core.env_server import Action, Observation

class SCAct(Action):
    t_id: int
    p_qty: int

class SCObs(Observation):
    t_cap: int
    p_ord: int
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}