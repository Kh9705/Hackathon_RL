from typing import Dict, Any, Optional
from openenv.core.env_server import Action, Observation

class SCAct(Action):
    """Supply chain action: order from supplier"""
    supplier_id: int = 0
    purchase_qty: int

class SCObs(Observation):
    """Supply chain observation: warehouse state"""
    pending_orders: int  # Orders waiting to be fulfilled
    warehouse_inventory: int  # Current stock level
    warehouse_capacity: int  # Max capacity
    demand_rate: float  # Units demanded per step
    stockout_cost: float = 0.0  # Cost from stockouts this step
    holding_cost: float = 0.0  # Cost from holding inventory
    info: Optional[Dict[str, Any]] = None

class SCSt(Observation):
    """Supply chain internal state (for state endpoint)"""
    episode_id: str
    step_count: int
    cumulative_reward: float
    episode_difficulty: str