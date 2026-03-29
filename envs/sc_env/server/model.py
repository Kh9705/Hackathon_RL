from dataclasses import dataclass
from typing import List
from core.env_server import Action, Observation, State

@dataclass
class SCAct(Action):
    t_id: int
    p_qty: int

@dataclass
class SCObs(Observation):
    wh_inv: int
    t_cap: List[int]
    p_ord: int
    d: bool
    r: float

@dataclass
class SCSt(State):
    e_id: str
    s_cnt: int