import random
from typing import Optional, Dict, Any

# Import OpenEnv base class
from openenv.core.env_server import Environment

from envs.sc_env.models import SCAct, SCObs, SCSt


class SCEnv(Environment):
    """
    Supply Chain Optimization Environment
    
    Simulates inventory management with:
    - Stochastic demand
    - Capacity constraints
    - Cost penalties (holding, stockout, ordering)
    """
    
    def __init__(self):
        """Initialize all environment state"""
        super().__init__()
        
        # Episode state
        self.episode_id: str = ""
        self.episode_difficulty: str = "easy"
        self.step_count: int = 0
        self.max_steps: int = 50
        
        # Supply chain state
        self.warehouse_inventory: int = 50
        self.warehouse_capacity: int = 100
        self.pending_orders: int = 0
        
        # Task-specific parameters
        self.base_demand: float = 10.0
        self.demand_variance: float = 5.0
        self.holding_cost: float = 0.5      # Cost per unit per step
        self.stockout_penalty: float = 10.0  # Penalty per unmet order
        self.order_cost: float = 1.0         # Cost per unit ordered
        
        # Metrics for reward calculation
        self.total_holding_cost: float = 0.0
        self.total_stockout: float = 0.0
        self.total_order_cost: float = 0.0
        self.cumulative_reward: float = 0.0
        
        # RNG for reproducibility
        self.rng: random.Random = random.Random()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> SCObs:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Format "task_name-variation" e.g., "supply_chain_easy-0"
            **kwargs: Additional OpenEnv framework parameters
        
        Returns:
            Initial observation
        """
        # Handle seed
        if seed is not None:
            self.rng.seed(seed)
        
        # Parse episode_id to determine difficulty
        self.episode_id = episode_id or "unknown"
        self._parse_difficulty_from_episode_id(episode_id)
        
        # Reset state based on difficulty
        self.step_count = 0
        self.warehouse_inventory = self._get_initial_inventory()
        self.pending_orders = self._get_initial_demand()
        
        # Reset metrics
        self.total_holding_cost = 0.0
        self.total_stockout = 0.0
        self.total_order_cost = 0.0
        self.cumulative_reward = 0.0
        
        # Return initial observation (NO reward/done in observation)
        return SCObs(
            pending_orders=self.pending_orders,
            warehouse_inventory=self.warehouse_inventory,
            warehouse_capacity=self.warehouse_capacity,
            demand_rate=self.base_demand,
            info={
                "step": self.step_count,
                "difficulty": self.episode_difficulty,
                "episode_id": self.episode_id
            }
        )

    def step(self, action: SCAct) -> tuple[SCObs, float, bool]:
        """
        Execute one step of the environment.
        
        Args:
            action: SCAct with supplier_id and purchase_qty
        
        Returns:
            (observation, reward, done)
            CRITICAL: OpenEnv framework expects these separately!
        """
        self.step_count += 1
        
        # 1. RECEIVE NEW ORDERS (stochastic demand)
        new_orders = self._generate_demand()
        self.pending_orders += new_orders
        
        # 2. FULFILL ORDERS FROM INVENTORY
        fulfilled_orders = min(self.pending_orders, self.warehouse_inventory)
        unmet_orders = self.pending_orders - fulfilled_orders
        self.warehouse_inventory -= fulfilled_orders
        self.pending_orders = unmet_orders
        
        # 3. APPLY ACTION (restock purchases)
        purchase_qty = action.purchase_qty
        purchase_qty = min(purchase_qty, self.warehouse_capacity - self.warehouse_inventory)
        purchase_qty = max(0, purchase_qty)  # Can't be negative
        
        self.warehouse_inventory += purchase_qty
        
        # 4. CALCULATE COSTS
        holding_cost = self.warehouse_inventory * self.holding_cost
        stockout_cost = unmet_orders * self.stockout_penalty
        order_cost = purchase_qty * self.order_cost
        
        step_cost = holding_cost + stockout_cost + order_cost
        
        # 5. CALCULATE REWARD
        # Higher is better: penalize costs, reward fulfillment
        fulfillment_rate = fulfilled_orders / (self.pending_orders + fulfilled_orders + 1e-6)
        step_reward = fulfillment_rate - (step_cost / 100.0)
        step_reward = max(-1.0, min(1.0, step_reward))  # Normalize to [-1, 1]
        
        self.cumulative_reward += step_reward
        self.total_holding_cost += holding_cost
        self.total_stockout += stockout_cost
        self.total_order_cost += order_cost
        
        # 6. CHECK TERMINATION
        done = self.step_count >= self.max_steps
        
        # 7. RETURN OBSERVATION (without reward/done)
        obs = SCObs(
            pending_orders=self.pending_orders,
            warehouse_inventory=self.warehouse_inventory,
            warehouse_capacity=self.warehouse_capacity,
            demand_rate=self.base_demand,
            info={
                "step": self.step_count,
                "fulfilled": fulfilled_orders,
                "unmet": unmet_orders,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "order_cost": order_cost
            }
        )
        
        return obs, step_reward, done

    def state(self) -> SCSt:
        """Get current internal state (for state endpoint)"""
        return SCSt(
            episode_id=self.episode_id,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            episode_difficulty=self.episode_difficulty
        )
    
    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> SCObs:
        """
        Async version of reset. Delegates to sync reset for now.
        """
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)
    
    async def step_async(self, action: SCAct) -> tuple[SCObs, float, bool]:
        """
        Async version of step. Delegates to sync step for now.
        """
        return self.step(action)
    
    def close(self):
        """
        Clean up resources (required by OpenEnv).
        """
        pass

    # ============ HELPER METHODS ============
    
    def _parse_difficulty_from_episode_id(self, episode_id: Optional[str]):
        """Extract difficulty level from episode_id string"""
        if not episode_id:
            self.episode_difficulty = "easy"
            return
        
        episode_id_lower = episode_id.lower()
        if "hard" in episode_id_lower:
            self.episode_difficulty = "hard"
            self._set_hard_difficulty()
        elif "medium" in episode_id_lower:
            self.episode_difficulty = "medium"
            self._set_medium_difficulty()
        else:
            self.episode_difficulty = "easy"
            self._set_easy_difficulty()
    
    def _set_easy_difficulty(self):
        """Easy: Low, predictable demand"""
        self.base_demand = 8.0
        self.demand_variance = 2.0
        self.warehouse_capacity = 150
        self.holding_cost = 0.3
        self.stockout_penalty = 5.0
        self.max_steps = 50
    
    def _set_medium_difficulty(self):
        """Medium: Moderate, variable demand"""
        self.base_demand = 15.0
        self.demand_variance = 5.0
        self.warehouse_capacity = 100
        self.holding_cost = 0.5
        self.stockout_penalty = 10.0
        self.max_steps = 50
    
    def _set_hard_difficulty(self):
        """Hard: High, volatile demand with capacity constraints"""
        self.base_demand = 25.0
        self.demand_variance = 10.0
        self.warehouse_capacity = 80
        self.holding_cost = 0.8
        self.stockout_penalty = 15.0
        self.max_steps = 50
    
    def _get_initial_inventory(self) -> int:
        """Set starting inventory based on difficulty"""
        if self.episode_difficulty == "easy":
            return 40
        elif self.episode_difficulty == "medium":
            return 30
        else:  # hard
            return 20
    
    def _get_initial_demand(self) -> int:
        """Set starting pending orders based on difficulty"""
        if self.episode_difficulty == "easy":
            return 5
        elif self.episode_difficulty == "medium":
            return 10
        else:  # hard
            return 20
    
    def _generate_demand(self) -> int:
        """Generate stochastic demand using Gaussian distribution"""
        demand = int(self.rng.gauss(self.base_demand, self.demand_variance))
        return max(0, demand)  # Can't have negative demand