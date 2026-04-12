from pydantic import BaseModel
from typing import Dict, Any, Optional
from openenv.core.env_server import Action, Observation


class SCAct(Action):
    """Action: Order a purchase quantity for a supplier"""
    supplier_id: int
    purchase_qty: int


class SCObs(Observation):
    """
    Observation returned after each step.
    IMPORTANT: Field names must match openenv.yaml observation_space exactly.
    Inherits 'done' and 'reward' from Observation base class.
    """
    pending_orders: int      # Number of pending customer orders
    warehouse_inventory: int # Current inventory level
    warehouse_capacity: int  # Maximum warehouse capacity
    demand_rate: float       # Orders per step (for next step prediction)
    info: Dict[str, Any] = {}  # Metadata


class SCSt(BaseModel):
    """Internal state for tracking"""
    episode_id: str
    step_count: int
    cumulative_reward: float = 0.0
    episode_difficulty: str = "easy"  # Track which task variant we're in


# Alternative: Pydantic models for stricter validation
# These can be used if the framework requires them

class SCActPydantic(BaseModel):
    supplier_id: int
    purchase_qty: int

    class Config:
        validate_assignment = True


class SCObsPydantic(BaseModel):
    pending_orders: int
    warehouse_inventory: int
    warehouse_capacity: int
    demand_rate: float
    info: Dict[str, Any] = {}

    class Config:
        validate_assignment = True