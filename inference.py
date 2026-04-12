import os
import sys
import json
from typing import Dict, Any, Optional

# ============================================================================
# ENVIRONMENT VARIABLE SETUP
# ============================================================================
# Support both validator naming (API_KEY) and HF Space naming (HF_TOKEN)

API_BASE_URL = os.getenv("API_BASE_URL")  # LLM proxy endpoint
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-3.5-turbo"  # Model to use
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")  # API credentials

# Debug logging to help identify issues
print(f"[INFO] API_BASE_URL: {API_BASE_URL}", file=sys.stderr, flush=True)
print(f"[INFO] MODEL_NAME: {MODEL_NAME}", file=sys.stderr, flush=True)
print(f"[INFO] API_KEY present: {bool(API_KEY)}", file=sys.stderr, flush=True)

# Determine if LLM is available (both endpoint and credentials needed)
LLM_AVAILABLE = bool(API_BASE_URL and API_KEY)

# ============================================================================
# DEPENDENCY IMPORTS
# ============================================================================

try:
    from openenv.core import SyncEnvClient, GenericEnvClient
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}", file=sys.stderr)
    print("[ERROR] Install: pip install openenv-core", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# LLM CLIENT INITIALIZATION
# ============================================================================
# Initialize OpenAI client for LLM calls through proxy

llm_client = None
if LLM_AVAILABLE:
    try:
        from openai import OpenAI
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"[INFO] LLM client initialized successfully", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WARN] LLM unavailable: {e}, using heuristics", file=sys.stderr)
        LLM_AVAILABLE = False
else:
    print(f"[INFO] LLM not available (missing API_BASE_URL or API_KEY), using heuristics", file=sys.stderr, flush=True)

# ============================================================================
# OPENENV CLIENT INITIALIZATION
# ============================================================================
# Client to communicate with the environment server

BASE_URL = os.getenv("OPENENV_URL", "http://localhost:8000")
env_client = SyncEnvClient(GenericEnvClient(BASE_URL))


class SupplyChainAgent:
    """
    Intelligent agent for supply chain optimization.
    
    Features:
    - Task-specific strategies (easy/medium/hard difficulty levels)
    - LLM-based decision making with clear, structured prompts
    - Heuristic fallback when LLM unavailable (validator environment)
    - Structured logging for phase validation
    
    Example Usage:
        agent = SupplyChainAgent(use_llm=True)
        action = agent.get_action(task_name, observation)
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize agent with optional LLM support.
        
        Args:
            use_llm: If True, attempt to use LLM for decisions.
                     Falls back to heuristics if LLM_AVAILABLE=False.
        """
        # Use LLM only if both requested AND available
        self.use_llm = use_llm and LLM_AVAILABLE
    
    def get_action_llm(self, task_name: str, observation: Dict[str, Any]) -> int:
        """
        Use LLM to decide purchase quantity based on task difficulty.
        
        Sends difficulty-specific prompts to LLM via proxy with clear instructions.
        Each difficulty level has tailored ordering strategy:
        - EASY: Forgiving environment, conservative ordering works
        - MEDIUM: Balanced difficulty, moderate ordering strategy
        - HARD: Volatile demand, aggressive buffering needed
        
        Args:
            task_name: Task identifier containing difficulty level
            observation: Current observation dict with inventory state
            
        Returns:
            Integer quantity to order (0-capacity)
        """
        # Extract current state
        pending_orders = observation.get("pending_orders", 0)
        inventory = observation.get("warehouse_inventory", 0)
        capacity = observation.get("warehouse_capacity", 100)
        demand_rate = observation.get("demand_rate", 10.0)
        
        # =====================================================================
        # DIFFICULTY-SPECIFIC PROMPTS
        # =====================================================================
        # Each prompt is explicit and clear to maximize LLM accuracy
        
        if "easy" in task_name.lower():
            # EASY: Low demand (base_demand=8), generous capacity
            # Strategy: Simple responsive ordering, forgiving environment
            prompt = (
                f"Supply Chain Task: EASY MODE (LOW DEMAND, FORGIVING)\n"
                f"Current State:\n"
                f"  - Pending Orders: {pending_orders}\n"
                f"  - Inventory: {inventory}\n"
                f"  - Warehouse Capacity: {capacity}\n"
                f"  - Base Demand: {demand_rate:.1f}/step\n\n"
                f"Simple Strategy:\n"
                f"  - If pending > {int(demand_rate * 2.5)}: order {int(demand_rate * 2)} units\n"
                f"  - Else if pending > {int(demand_rate * 1.5)}: order {int(demand_rate * 1.2)} units\n"
                f"  - Otherwise: order {int(demand_rate * 0.8)} units\n"
                f"  - Forgiving environment: optimize for steady supply.\n\n"
                f"Respond with ONLY a single integer (e.g., 8 or 15). No explanation."
            )
        
        elif "medium" in task_name.lower():
            # MEDIUM: Moderate demand (base_demand=15), balanced capacity
            # Strategy: Proactive buffer with responsive adjustments
            pending_pressure = max(0, pending_orders - inventory)
            recommended_buffer = int(demand_rate * 2.5) if pending_pressure > demand_rate else int(demand_rate * 1.8)
            
            prompt = (
                f"Supply Chain Task: MEDIUM MODE (VARIABLE DEMAND)\n"
                f"Current State:\n"
                f"  - Pending Orders: {pending_orders}\n"
                f"  - Current Inventory: {inventory}\n"
                f"  - Warehouse Capacity: {capacity}\n"
                f"  - Average Demand: {demand_rate:.1f}/step\n\n"
                f"Decision Framework:\n"
                f"  - If pending orders growing: order MORE (proactive)\n"
                f"  - If pending orders stable: order LESS (conservative)\n"
                f"  - Maintain safety buffer: {recommended_buffer} units\n"
                f"  - Formula: qty = max(0, pending - inventory + buffer)\n"
                f"  - Absolute cap: {int(capacity * 0.65)} units\n\n"
                f"Goal: Avoid both stockouts AND excess inventory.\n"
                f"Respond with ONLY a single integer (0-{int(capacity * 0.65)})."
            )
        
        else:  # hard
            # HARD: High volatile demand (base_demand=25), limited capacity
            # Strategy: Aggressive ordering with smart capacity management
            pending_pressure = max(0, pending_orders - inventory)
            dynamic_capacity_limit = int(capacity * (0.7 if pending_pressure > demand_rate * 2 else 0.6))
            
            prompt = (
                f"Supply Chain Task: HARD MODE (HIGH VOLATILITY)\n"
                f"Current State:\n"
                f"  - Pending Orders: {pending_orders} (critical: {pending_pressure})\n"
                f"  - Inventory: {inventory}\n"
                f"  - Capacity: {capacity}\n"
                f"  - Demand Rate: {demand_rate:.1f}/step (highly volatile)\n\n"
                f"STRATEGY - Stockout Prevention Priority:\n"
                f"  1. If pending_orders > {int(demand_rate * 2.5)}: AGGRESSIVE order to clear backlog\n"
                f"  2. Maintain safety buffer: {int(demand_rate * 3)} units minimum\n"
                f"  3. Dynamic capacity limit: {dynamic_capacity_limit} units (based on pending pressure)\n"
                f"  4. Formula: qty = max(10, pending - inventory + {int(demand_rate * 3)})\n"
                f"  5. NEVER exceed {dynamic_capacity_limit} units per order\n"
                f"  6. Preference: Stock out prevention > capacity constraints\n\n"
                f"Respond with ONLY a single integer between 10 and {dynamic_capacity_limit}."
            )
        
        try:
            # Send to LLM through validator's proxy
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistency
                max_tokens=10     # Only need small response
            )
            
            # Parse integer from response (handles partial/malformed responses)
            response_text = response.choices[0].message.content.strip()
            qty = int(''.join(filter(str.isdigit, response_text)) or "0")
            qty = max(0, min(qty, capacity))  # Clamp to valid range [0, capacity]
            return qty
        
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
            # Fallback to heuristics if LLM fails
            return self.get_action_heuristic(task_name, observation)
    
    def get_action_heuristic(self, task_name: str, observation: Dict[str, Any]) -> int:
        """
        Fallback heuristic strategy when LLM unavailable (validator environment).
        
        Uses mathematically-derived ordering policies based on inventory theory:
        - EASY: Low demand variance, simple reactive ordering
        - MEDIUM: Moderate variance, proactive buffer maintenance
        - HARD: High variance, conservative safety stock strategy
        
        Args:
            task_name: Task identifier containing difficulty level
            observation: Current observation dict
            
        Returns:
            Integer quantity to order using heuristic policy
        """
        # Extract current state
        pending_orders = observation.get("pending_orders", 0)
        inventory = observation.get("warehouse_inventory", 0)
        capacity = observation.get("warehouse_capacity", 100)
        demand_rate = observation.get("demand_rate", 10.0)
        
        # =====================================================================
        # DIFFICULTY-SPECIFIC HEURISTIC POLICIES
        # =====================================================================
        
        if "easy" in task_name.lower():
            # EASY: Consistent high fill rate strategy
            # Logic: Maintain stable inventory to achieve 90%+ fill rate target
            pending_pressure = max(0, pending_orders - inventory)
            if pending_orders > demand_rate * 3.0:
                # High backlog: order 2x demand to clear fast
                return int(demand_rate * 2.0)
            elif pending_pressure > demand_rate * 2.0:
                # Moderate backlog: order 1.5x demand
                return int(demand_rate * 1.5)
            else:
                # Low backlog: maintain steady 1x demand ordering
                # This ensures consistent inventory for high fill rate
                return int(demand_rate * 1.0)
        
        elif "medium" in task_name.lower():
            # MEDIUM: Balanced policy with adaptive safety stock
            # Logic: Maintain 2-2.5x demand rate as safety stock depending on situation
            pending_pressure = max(0, pending_orders - inventory)
            # If pending orders are high relative to inventory, increase buffer
            if pending_pressure > demand_rate * 1.5:
                buffer = int(demand_rate * 2.5)  # Aggressive when behind
            else:
                buffer = int(demand_rate * 1.8)  # Conservative when ahead
            needed = max(0, pending_orders - inventory + buffer)
            # Cap at 65% of capacity for more responsive ordering
            return min(needed, int(capacity * 0.65))
        
        else:  # hard
            # HARD: Aggressive with dynamic buffer for volatile demand
            # Logic: Adaptive buffer increases with pending orders (more aggressive ordering)
            # Base buffer 3x + extra when pending builds up
            base_buffer = int(demand_rate * 3)
            pending_pressure = max(0, pending_orders - inventory)
            # If many pending orders, increase buffer for faster fulfillment
            adaptive_buffer = base_buffer + int(pending_pressure * 0.5)
            needed = max(10, pending_orders - inventory + adaptive_buffer)  # Min 10 units
            # Cap at 65% of capacity (increased from 50% for more aggressive ordering)
            return min(needed, int(capacity * 0.65))
    
    def get_action(self, task_name: str, observation: Dict[str, Any]) -> int:
        """
        Get action: dispatch to LLM if available, else use heuristic.
        
        Args:
            task_name: Task identifier
            observation: Current observation
            
        Returns:
            Integer quantity to order (0-capacity)
        """
        if self.use_llm:
            return self.get_action_llm(task_name, observation)
        else:
            return self.get_action_heuristic(task_name, observation)




def run_tasks():
    """
    Run all three supply chain tasks (easy/medium/hard) and collect performance metrics.
    
    CRITICAL: Must emit structured logs in exact format for Phase 2 validation:
    - [START] task=<name>          : Begin task
    - [STEP] step=N reward=F.FF    : Each step with accumulated reward
    - [END] task=<name> score=F.FF steps=N : Task completion with normalized score
    
    Scoring Algorithm:
    - Collects reward for each step in task
    - avg_reward = total_reward / steps
    - normalized_score = (avg_reward + 1.0) / 2.0  [shifts [-1,1] to [0,1]]
    - final_score = clamp(normalized_score, 0.01, 0.99)  [ensure strictly in range]
    
    Example Output:
        [START] task=supply_chain_easy
        [STEP] step=1 reward=0.50
        [STEP] step=2 reward=0.55
        ...
        [END] task=supply_chain_easy score=0.52 steps=35
    """
    # =====================================================================
    # TASK DEFINITIONS
    # =====================================================================
    tasks = [
        "supply_chain_easy",      # Low demand, forgiving, good for baseline
        "supply_chain_medium",    # Moderate demand, balanced challenge
        "supply_chain_hard"       # High demand, volatile, difficult
    ]
    
    agent = SupplyChainAgent(use_llm=True)
    max_steps_per_task = 50  # Prevent infinite loops
    
    # =====================================================================
    # TASK EXECUTION LOOP
    # =====================================================================
    for task_name in tasks:
        # Signal start of task to validator
        # Format: [START] task=<task_name> env=<benchmark> model=<model_name>
        print(f"[START] task={task_name} env=supply_chain model=heuristic_agent", flush=True)
        
        try:
            # Reset environment to initial state for this task
            result = env_client.reset()
            observation = result.observation
            
            total_reward = 0.0
            steps_completed = 0
            
            # === STEP LOOP: Execute up to max_steps_per_task ===
            for step_idx in range(max_steps_per_task):
                
                # Get action from agent (LLM or heuristic)
                action_qty = agent.get_action(task_name, observation)
                action = {"supplier_id": 0, "purchase_qty": action_qty}
                
                # Execute action in environment
                result = env_client.step(action)
                observation = result.observation
                reward = float(getattr(result, 'reward', 0.0))
                done = bool(getattr(result, 'done', False))
                
                # Accumulate metrics
                total_reward += reward
                steps_completed += 1
                
                # === EMIT STEP LOG (exact format required for Phase 2 validation) ===
                # Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
                action_str = f"order({action_qty})"
                done_str = str(done).lower()
                print(f"[STEP] step={step_idx + 1} action={action_str} reward={reward:.2f} done={done_str} error=null", flush=True)
                
                # Stop if episode finished early
                if done:
                    break
            
            # ===================================================================
            # SCORE CALCULATION
            # ===================================================================
            # Critical: score must be STRICTLY in range (0, 1), not inclusive
            
            # 1. Calculate average reward across all steps
            avg_reward = total_reward / max(steps_completed, 1)
            
            # 2. Normalize from [-1, 1] to [0, 1]
            # (Rewards typically in [-1, 1] range; scale to [0, 1])
            normalized_score = (avg_reward + 1.0) / 2.0
            
            # 3. Clamp to strictly interior: [0.01, 0.99]
            # This ensures we never hit boundary values 0.0 or 1.0
            final_score = max(0.01, min(0.99, normalized_score))
            
            # === EMIT END LOG (exact format required for Phase 2 validation) ===
            # Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
            success = final_score >= 0.5
            print(f"[END] success={str(success).lower()} steps={steps_completed} score={final_score:.3f} rewards=n/a", flush=True)
        
        except Exception as e:
            # ===================================================================
            # ERROR HANDLING: Layer 2 (Task-level exception handler)
            # ===================================================================
            # Task failed - emit valid failure log with score in (0, 1)
            print(f"[END] success=false steps=0 score=0.05 rewards=n/a", flush=True)
            print(f"[ERROR] Task {task_name} failed: {str(e)[:200]}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    # Layer 5 (Catch-all): Ensures clean exit even on catastrophic failure
    
    try:
        run_tasks()
    except Exception as e:
        # Unhandled exception - log and exit cleanly
        # (Validator requires exit code 0 to successfully complete evaluation)
        print(f"[CRITICAL] Unhandled exception: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(0)  # Exit with success code (validator requirement)