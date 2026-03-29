from pydantic import BaseModel
from typing import Dict, Any

class SCAct(BaseModel):
    t_id: int
    p_qty: int

class SCObs(BaseModel):
    t_cap: int
    p_ord: int
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}