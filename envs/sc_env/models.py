from typing import Dict, Any, Optional
from openenv.core.env_server import Action, Observation


# ============================================================================
# ACTION MODEL: What agent can do each step
# ============================================================================

class SCAct(Action):
    """
    Supply Chain Action: Order from supplier
    
    Agent sends this action each step to control inventory through ordering.
    
    Attributes:
        supplier_id (int): Which supplier to order from (default: 0, single supplier)
        purchase_qty (int): Quantity to order (units). Default: 0 (no order)
        
    Example:
        action = SCAct(supplier_id=0, purchase_qty=50)
        # Orders 50 units from supplier 0
    """
    supplier_id: int = 0           # Single supplier for simplicity
    purchase_qty: int = 0          # Default to 0 if not specified (no order)


# ============================================================================
# OBSERVATION MODEL: What agent observes each step
# ============================================================================

class SCObs(Observation):
    """
    Supply Chain Observation: Current warehouse inventory state
    
    Includes warehouse inventory metrics and cost signals for the current step.
    Agent uses this to make ordering decisions.
    
    Attributes:
        pending_orders (int): Orders waiting to be fulfilled from inventory
        warehouse_inventory (int): Current stock level in warehouse
        warehouse_capacity (int): Maximum capacity of warehouse
        demand_rate (float): Stochastic demand units per step (informational)
        stockout_cost (float): Cost incurred from stockouts this step
        holding_cost (float): Cost incurred from holding inventory this step
        info (dict): Optional metadata dictionary for debugging
        
    Example:
        obs = SCObs(
            pending_orders=10,
            warehouse_inventory=25,
            warehouse_capacity=100,
            demand_rate=8.5,
            stockout_cost=5.0,
            holding_cost=2.5,
            info={"step": 5}
        )
    """
    pending_orders: int            # Orders waiting to be fulfilled
    warehouse_inventory: int       # Current stock level
    warehouse_capacity: int        # Max capacity
    demand_rate: float             # Units demanded per step (informational)
    stockout_cost: float = 0.0     # Cost from stockouts this step
    holding_cost: float = 0.0      # Cost from holding inventory
    info: Optional[Dict[str, Any]] = None  # Optional metadata


# ============================================================================
# STATE MODEL: Internal environment state (for debugging)
# ============================================================================

class SCSt(Observation):
    """
    Supply Chain Internal State: Episode metadata
    
    Returned by environment.state() endpoint for debugging/inspection.
    Contains internal simulation state not available in observations.
    
    Attributes:
        episode_id (str): Episode identifier (used to determine difficulty)
        step_count (int): Current step within episode
        cumulative_reward (float): Sum of rewards so far in episode
        episode_difficulty (str): Parsed difficulty level (easy/medium/hard)
        
    Example:
        state = SCSt(
            episode_id="supply_chain_easy_001",
            step_count=15,
            cumulative_reward=8.5,
            episode_difficulty="easy"
        )
    """
    episode_id: str                # Episode identifier
    step_count: int                # Current step
    cumulative_reward: float       # Sum of rewards
    episode_difficulty: str        # Parsed difficulty (easy/medium/hard)