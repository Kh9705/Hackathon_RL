import random
import math
from typing import Optional, Dict, Any, List

# Import OpenEnv base class
from openenv.core.env_server import Environment

from .model import SCAct, SCObs, SCSt


class SCEnv(Environment):
    """
    Advanced Supply Chain Optimization Environment for RL Training
    
    A realistic multi-supplier inventory management simulator featuring:
    
    Core Features:
    - Multi-supplier selection (3 suppliers with different characteristics)
    - Lead time delays (orders arrive after 1-3 steps)
    - Seasonal demand patterns (sinusoidal variation)
    - Demand shocks (COVID-style disruptions)
    - Advanced multi-objective reward structure
    
    Agent Goal:
    - Minimize total costs while maintaining service level
    - Make supplier selection decisions based on lead times/costs
    - Adapt to demand volatility and shocks
    - Balance inventory efficiently
    
    Features:
    - Stochastic demand with seasonal patterns
    - Multiple difficulty levels (easy/medium/hard)
    - Realistic cost penalties and reward signals
    - Deterministic with seed control for reproducibility
    
    ADVANCED REWARD DESIGN:
    
    The environment uses a multi-objective reward structure that combines four key components:
    
    1. SERVICE LEVEL REWARD (50% weight):
       - Measures fill rate (fulfilled_demand / total_demand)
       - Target: 85%+ fill rate for good performance
       - Calculation: (fill_rate - 0.7) * 2 (normalized to roughly [-0.6, +1.0])
       - Real-world: Directly impacts customer satisfaction and revenue
    
    2. COST EFFICIENCY REWARD (30% weight):
       - Tracks cost per unit delivered: total_cost / units_fulfilled
       - Target: <$1.2 per unit for good performance
       - Calculation: 1.0 - (cost_per_unit / target_cost) clamped to [-1, 1]
       - Real-world: Direct profit margin indicator
    
    3. SUSTAINABILITY REWARD (10% weight):
       - Bonus for using cheaper, more efficient suppliers (lower cost = greener)
       - Rewards consolidation (fewer orders = lower carbon footprint)
       - Calculation: Bonus for achieving >80% fill with <$1.2 cost per unit
       - Real-world: ESG metrics and environmental responsibility
    
    4. EFFICIENCY REWARD (10% weight):
       - Penalizes excessive inventory holding
       - Rewards efficient capital utilization
       - Calculation: -inventory_holding_cost / warehouse_capacity
       - Real-world: Working capital optimization and ROI
    
    Mathematical Formula:
    $$R = 0.50 \times R_{service} + 0.30 \times R_{cost} + 0.10 \times R_{sustainability} + 0.10 \times R_{efficiency}$$
    
    Reward Dynamics:
    - POSITIVE scenarios: High fill rate + low cost + lean inventory → R > 0.7
    - NEUTRAL scenarios: Meeting minimum thresholds → R ≈ 0.0
    - NEGATIVE scenarios: Stockouts or excessive costs → R < -0.5
    
    The weighted combination ensures agents balance multiple objectives rather than gaming
    a single metric, reflecting real supply chain management complexity.
    
    Example:
        >>> env = SCEnv()
        >>> obs = env.reset(episode_id="supply_chain_easy-0")
        >>> action = SCAct(supplier_id=1, purchase_qty=25)  # Select supplier 1
        >>> obs, reward, done = env.step(action)
    """
    
    def __init__(self):
        """
        Initialize environment state variables for advanced supply chain simulation.
        
        Sets default values for inventory, multi-supplier system, demand parameters.
        """
        super().__init__()
        
        # === EPISODE CONTROL ===
        self.episode_id: str = ""
        self.episode_difficulty: str = "easy"
        self.step_count: int = 0
        self.max_steps: int = 50
        
        # === INVENTORY STATE ===
        self.warehouse_inventory: int = 50
        self.warehouse_capacity: int = 100
        self.pending_orders: int = 0
        
        # === MULTI-SUPPLIER SYSTEM ===
        # Supplier 0: Fast (lead_time=1), high cost ($1.5/unit)
        # Supplier 1: Balanced (lead_time=2), medium cost ($1.0/unit)
        # Supplier 2: Cheap (lead_time=3), low cost ($0.6/unit)
        self.suppliers = {
            0: {"name": "Express", "lead_time": 1, "cost": 1.5},
            1: {"name": "Standard", "lead_time": 2, "cost": 1.0},
            2: {"name": "Economy", "lead_time": 3, "cost": 0.6},
        }
        self.pending_deliveries: List[Dict[str, int]] = []  # [{delivery_step: int, quantity: int, supplier_id: int}]
        
        # === DEMAND DYNAMICS ===
        self.base_demand: float = 10.0
        self.demand_variance: float = 5.0
        self.seasonal_amplitude: float = 3.0  # Seasonal variation amplitude
        self.demand_shock_probability: float = 0.0  # Will be set by difficulty
        self.current_demand_shock: float = 1.0  # Multiplier for demand (1.0 = normal, 2.0 = crisis)
        
        # === COST STRUCTURE ===
        self.holding_cost_per_unit: float = 0.5
        self.stockout_penalty: float = 10.0
        self.sustainability_cost: float = 0.1  # Penalty for high inventory (environmental)
        self.service_level_target: float = 0.95  # 95% fulfillment target
        
        # === METRICS FOR LEARNING ===
        self.total_holding_cost: float = 0.0
        self.total_stockout: float = 0.0
        self.total_order_cost: float = 0.0
        self.total_sustainability_cost: float = 0.0
        self.cumulative_reward: float = 0.0
        self.service_level: float = 1.0
        self.fulfilled_orders: int = 0
        self.total_demand: int = 0
        
        # === RANDOMNESS ===
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
        
        # Return initial observation
        return SCObs(
            pending_orders=self.pending_orders,
            warehouse_inventory=self.warehouse_inventory,
            warehouse_capacity=self.warehouse_capacity,
            demand_rate=self.base_demand,
            reward=0.0,
            done=False,
            info={
                "step": self.step_count,
                "difficulty": self.episode_difficulty,
                "episode_id": self.episode_id
            }
        )

    def step(self, action: SCAct) -> tuple[SCObs, float, bool]:
        """
        Execute one step of the environment with advanced supply chain mechanics.
        
        New Features:
        - Multi-supplier lead times: Orders don't arrive instantly
        - Demand shocks: COVID-style disruptions
        - Seasonal patterns: Cyclic demand variation
        - Sustainability costs: Reward for lower inventory
        
        Args:
            action: SCAct with supplier_id and purchase_qty
        
        Returns:
            (observation, reward, done)
        """
        self.step_count += 1
        
        # === STEP 1: PROCESS DELIVERIES (Lead time) ===
        # Orders from previous steps arrive based on lead times
        new_pending_deliveries = []
        for delivery in self.pending_deliveries:
            delivery["arrival_step"] -= 1
            if delivery["arrival_step"] <= 0:
                # Delivery arrived
                self.warehouse_inventory += delivery["quantity"]
            else:
                # Still in transit
                new_pending_deliveries.append(delivery)
        self.pending_deliveries = new_pending_deliveries
        
        # === STEP 2: GENERATE DEMAND (with seasonality + shocks) ===
        new_orders = self._generate_demand_advanced()
        self.pending_orders += new_orders
        self.total_demand += new_orders
        
        # === STEP 3: FULFILL ORDERS FROM CURRENT INVENTORY ===
        fulfilled_orders = min(self.pending_orders, self.warehouse_inventory)
        unmet_orders = self.pending_orders - fulfilled_orders
        self.warehouse_inventory -= fulfilled_orders
        self.pending_orders = unmet_orders
        self.fulfilled_orders += fulfilled_orders
        
        # === STEP 4: PROCESS ORDER ACTION (Multi-supplier) ===
        supplier_id = action.supplier_id if action.supplier_id in self.suppliers else 1
        purchase_qty = action.purchase_qty
        purchase_qty = min(purchase_qty, self.warehouse_capacity - self.warehouse_inventory)
        purchase_qty = max(0, purchase_qty)
        
        if purchase_qty > 0:
            # Create delayed delivery based on supplier lead time
            supplier = self.suppliers[supplier_id]
            delivery = {
                "quantity": purchase_qty,
                "arrival_step": supplier["lead_time"],
                "supplier_id": supplier_id,
                "cost_per_unit": supplier["cost"]
            }
            self.pending_deliveries.append(delivery)
        
        # === STEP 5: CALCULATE ADVANCED COSTS ===
        holding_cost = self.warehouse_inventory * self.holding_cost_per_unit
        stockout_cost = unmet_orders * self.stockout_penalty
        
        # Order cost depends on supplier selection
        order_cost = 0
        for delivery in self.pending_deliveries:
            if delivery["arrival_step"] == self.suppliers[delivery["supplier_id"]]["lead_time"]:
                # This is a newly placed order
                order_cost += delivery["quantity"] * self.suppliers[delivery["supplier_id"]]["cost"]
        
        # Sustainability cost: penalize high inventory
        sustainability_cost = max(0, self.warehouse_inventory - 50) * self.sustainability_cost
        
        total_cost = holding_cost + stockout_cost + order_cost + sustainability_cost
        
        # === STEP 6: CALCULATE ADVANCED REWARD ===
        # Multi-objective: service level + cost efficiency
        if self.total_demand > 0:
            current_service_level = self.fulfilled_orders / self.total_demand
        else:
            current_service_level = 1.0
        
        # Reward structure - adjusted by difficulty:
        # - Bonus for high service level (meeting orders)
        # - Penalty for high costs (scaled by difficulty)
        # - Bonus for sustainability
        service_bonus = current_service_level * 0.5  # [0, 0.5]
        
        # Difficulty-adjusted cost penalty (easier tasks should reward more generously)
        if "easy" in self.episode_difficulty.lower():
            cost_penalty = -(total_cost / 150.0)  # Generous scaling for easy
        elif "medium" in self.episode_difficulty.lower():
            cost_penalty = -(total_cost / 120.0)  # Moderate scaling for medium
        else:  # hard
            cost_penalty = -(total_cost / 100.0)  # Strict scaling for hard
            
        sustainability_bonus = -max(0, self.warehouse_inventory - 40) * 0.01  # Reward lower inventory
        
        step_reward = service_bonus + cost_penalty + sustainability_bonus
        step_reward = max(-1.0, min(1.0, step_reward))  # Clamp to [-1, 1]
        
        self.cumulative_reward += step_reward
        self.total_holding_cost += holding_cost
        self.total_stockout += stockout_cost
        self.total_order_cost += order_cost
        self.total_sustainability_cost += sustainability_cost
        self.service_level = current_service_level
        
        # === STEP 7: CHECK TERMINATION ===
        done = self.step_count >= self.max_steps
        
        # === STEP 8: RETURN OBSERVATION ===
        obs = SCObs(
            pending_orders=self.pending_orders,
            warehouse_inventory=self.warehouse_inventory,
            warehouse_capacity=self.warehouse_capacity,
            demand_rate=self.base_demand * self.current_demand_shock,
            reward=step_reward,
            done=done,
            info={
                "step": self.step_count,
                "fulfilled": fulfilled_orders,
                "unmet": unmet_orders,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "order_cost": order_cost
            }
        )
        
        return obs

    def state(self) -> SCSt:
        """Get current internal state (for state endpoint)"""
        return SCSt(
            episode_id=self.episode_id,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            episode_difficulty=self.episode_difficulty
        )
    
    # =========================================================================
    # INDUSTRY-STANDARD METRICS & GRADING
    # =========================================================================
    
    def calculate_industry_metrics(self) -> Dict[str, float]:
        """
        Calculate supply chain industry-standard KPIs.
        
        Real supply chain managers track these metrics:
        - Fill Rate: % of demand fulfilled (target: 95%+)
        - Inventory Turns: How many times inventory cycles
        - Days of Supply: How many days of demand in inventory
        - Cost per Unit: Unit economics
        
        Returns:
            Dict with all industry metrics
        """
        # 1. FILL RATE (% of demand fulfilled)
        if self.total_demand > 0:
            fill_rate = self.fulfilled_orders / self.total_demand
        else:
            fill_rate = 1.0
        
        # 2. PERFECT ORDER RATE (on-time deliveries)
        on_time_deliveries = sum(1 for d in self.pending_deliveries 
                                 if d.get("arrival_step", 0) <= 1)
        if self.total_demand > 0:
            perfect_order_rate = on_time_deliveries / self.total_demand
        else:
            perfect_order_rate = 1.0
        
        # 3. INVENTORY TURNS
        avg_inventory = max(self.warehouse_inventory, 1)
        inventory_turns = self.total_demand / avg_inventory if avg_inventory > 0 else 0
        
        # 4. DAYS OF SUPPLY
        daily_demand = self.base_demand
        days_supply = self.warehouse_inventory / max(daily_demand, 1)
        
        # 5. COST OF GOODS SOLD
        cogs = self.total_order_cost
        
        # 6. CASH FLOW IMPACT
        cash_impact = self.total_holding_cost + self.total_order_cost
        
        # 7. COST PER UNIT FULFILLED
        cost_per_unit = cogs / max(self.fulfilled_orders, 1)
        
        return {
            "fill_rate": min(1.0, fill_rate),
            "perfect_order_rate": min(1.0, perfect_order_rate),
            "inventory_turns": inventory_turns,
            "days_supply": days_supply,
            "cogs": cogs,
            "cash_impact": cash_impact,
            "cost_per_unit": cost_per_unit,
            "total_demand": self.total_demand,
            "total_fulfilled": self.fulfilled_orders
        }
    
    def grade_episode(self) -> float:
        """
        Grade the entire episode based on task difficulty.
        Routes to task-specific grader.
        """
        if "easy" in self.episode_difficulty.lower():
            return self.grade_easy_task()
        elif "medium" in self.episode_difficulty.lower():
            return self.grade_medium_task()
        else:
            return self.grade_hard_task()

    def grade_easy_task(self) -> float:
        """
        Grade EASY task: Forgiving environment.
        Focus: Learn basic ordering principles.
        Target: 90%+ fill rate.
        """
        fill_rate = self.fulfilled_orders / max(self.total_demand, 1)
        
        if fill_rate >= 0.90:
            service_score = 1.0
        elif fill_rate >= 0.80:
            service_score = 0.8
        elif fill_rate >= 0.70:
            service_score = 0.6
        else:
            service_score = 0.4
        
        cost_efficiency = 1.0 - (self.total_order_cost / 500.0)
        cost_score = max(0.5, min(1.0, cost_efficiency))
        
        grade = 0.7 * service_score + 0.3 * cost_score
        
        if cost_efficiency > 0.8:
            grade += 0.05
        
        return max(0.01, min(0.99, grade))

    def grade_medium_task(self) -> float:
        """
        Grade MEDIUM task: Balanced difficulty.
        Focus: Balance service level and cost control.
        Target: 80-85% fill rate with cost optimization.
        """
        fill_rate = self.fulfilled_orders / max(self.total_demand, 1)
        
        if fill_rate >= 0.85:
            service_score = 1.0
        elif fill_rate >= 0.75:
            service_score = 0.8
        elif fill_rate >= 0.65:
            service_score = 0.6
        else:
            service_score = 0.3
        
        cost_per_unit_filled = self.total_order_cost / max(self.fulfilled_orders, 1)
        
        if cost_per_unit_filled <= 1.2:
            cost_score = 1.0
        elif cost_per_unit_filled <= 1.6:
            cost_score = 0.8
        elif cost_per_unit_filled <= 2.0:
            cost_score = 0.5
        else:
            cost_score = 0.3
        
        grade = 0.5 * service_score + 0.5 * cost_score
        
        if fill_rate < 0.5 and cost_per_unit_filled < 1.0:
            grade *= 0.7
        
        return max(0.01, min(0.99, grade))

    def grade_hard_task(self) -> float:
        """
        Grade HARD task: Frontier challenge.
        Focus: Handle volatility, tight capacity, frequent shocks.
        Target: 70-75% fill rate with optimized costs.
        """
        fill_rate = self.fulfilled_orders / max(self.total_demand, 1)
        
        if fill_rate >= 0.80:
            service_score = 1.0
        elif fill_rate >= 0.70:
            service_score = 0.9
        elif fill_rate >= 0.60:
            service_score = 0.7
        elif fill_rate >= 0.50:
            service_score = 0.5
        else:
            service_score = 0.2
        
        cost_per_unit = self.total_order_cost / max(self.fulfilled_orders, 1)
        
        if cost_per_unit <= 1.5:
            cost_score = 1.0
        elif cost_per_unit <= 1.8:
            cost_score = 0.85
        elif cost_per_unit <= 2.0:
            cost_score = 0.7
        elif cost_per_unit <= 2.5:
            cost_score = 0.5
        else:
            cost_score = 0.3
        
        grade = 0.6 * service_score + 0.4 * cost_score
        
        if self.current_demand_shock > 1.5 and fill_rate >= 0.7:
            grade += 0.1
        
        return max(0.01, min(0.99, grade))
    
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
        self.seasonal_amplitude = 1.0  # Minimal seasonality
        self.demand_shock_probability = 0.0  # No shocks
        self.warehouse_capacity = 150
        self.holding_cost = 0.3
        self.stockout_penalty = 5.0
        self.max_steps = 50
    
    def _set_medium_difficulty(self):
        """Medium: Moderate, variable demand"""
        self.base_demand = 15.0
        self.demand_variance = 5.0
        self.seasonal_amplitude = 2.0  # Moderate seasonality
        self.demand_shock_probability = 0.03  # 3% chance of shock per step
        self.warehouse_capacity = 100
        self.holding_cost = 0.5
        self.stockout_penalty = 10.0
        self.max_steps = 50
    
    def _set_hard_difficulty(self):
        """Hard: High, volatile demand with capacity constraints"""
        self.base_demand = 25.0
        self.demand_variance = 10.0
        self.seasonal_amplitude = 4.0  # Strong seasonality
        self.demand_shock_probability = 0.08  # 8% chance of shock per step
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
    
    def _generate_demand_advanced(self) -> int:
        """
        Generate advanced demand with:
        - Seasonal patterns (sinusoidal variation)
        - Random demand shocks (COVID-style disruptions)
        - Stochastic noise
        """
        # === SEASONAL COMPONENT ===
        # Demand varies sinusoidally over ~30-step cycle
        seasonal_factor = 1.0 + self.seasonal_amplitude * 0.1 * math.sin(2 * math.pi * self.step_count / 30.0)
        
        # === DEMAND SHOCK COMPONENT ===
        # Random 5% chance of demand shock at each step (for hard mode)
        if self.rng.random() < self.demand_shock_probability:
            self.current_demand_shock = self.rng.uniform(1.5, 2.5)  # Demand spike
        else:
            # Gradually return to normal after shock
            self.current_demand_shock = self.current_demand_shock * 0.9 + 1.0 * 0.1
        
        # === COMBINE COMPONENTS ===
        base_with_shock = self.base_demand * self.current_demand_shock * seasonal_factor
        stochastic_demand = int(self.rng.gauss(base_with_shock, self.demand_variance))
        
        return max(0, stochastic_demand)
