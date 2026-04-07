import os
import time
import json
from openai import OpenAI
from openenv.core import SyncEnvClient, GenericEnvClient

# Hackathon required variables
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the OpenAI Client (Mandatory per checklist)
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Initialize the OpenEnv Client
BASE_URL = os.getenv("OPENENV_URL", "http://0.0.0.0:8000")
base_env_client = GenericEnvClient(BASE_URL)
env_client = SyncEnvClient(base_env_client)

def parse_observation(obs):
    """Extract structured data from observation"""
    try:
        if isinstance(obs, str):
            # Try to parse JSON-like observation
            import re
            p_ord = float(re.search(r'p_ord["\']?\s*:\s*(\d+)', str(obs)).group(1)) if re.search(r'p_ord', str(obs)) else 0
            inventory = float(re.search(r'inventory["\']?\s*:\s*(\d+)', str(obs)).group(1)) if re.search(r'inventory', str(obs)) else 0
            return {"p_ord": p_ord, "inventory": inventory}
        elif isinstance(obs, dict):
            return obs
        return {"p_ord": 0, "inventory": 0}
    except:
        return {"p_ord": 0, "inventory": 0}

def get_strategy_prompt(task_name, obs_data):
    """Generate task-specific prompts for different difficulty levels"""
    p_ord = obs_data.get("p_ord", 0)
    inventory = obs_data.get("inventory", 0)
    
    if "simple" in task_name.lower():
        return (f"Supply chain state: pending_orders={p_ord}, inventory={inventory}. "
                f"Decision: If pending_orders > 40, order 25 units. Else order 8 units. "
                f"Return ONLY the order quantity as a number.")
    
    elif "medium" in task_name.lower():
        return (f"Supply chain management: pending_orders={p_ord}, current_inventory={inventory}. "
                f"Apply intelligent ordering: order qty = max(pending_orders * 0.6, 10), min(50). "
                f"Consider: high pending = buy more, low inventory = buy more. "
                f"Return ONLY the integer order quantity.")
    
    else:  # hard
        return (f"Advanced supply chain optimization: pending_orders={p_ord}, inventory={inventory}. "
                f"Implement demand-aware strategy: "
                f"1. If p_ord > 60: aggressive order (pending * 0.7) "
                f"2. If 20 < p_ord <= 60: moderate (pending * 0.5) "
                f"3. If p_ord <= 20: conservative + buffer (15 units) "
                f"4. Cap orders at 50 max. "
                f"Return ONLY the calculated order quantity as integer.")

def run_agent():
    tasks = ["supply_chain_simple", "supply_chain_medium", "supply_chain_hard"]
    max_steps_per_task = 15  # Increased for better score accumulation
    
    for task_name in tasks:
        print(f"[START] task={task_name}", flush=True)
        
        try:
            result = env_client.reset()
            obs = result.observation
            total_reward = 0.0
            steps_completed = 0
            max_reward_seen = 0.0
            
            for step in range(max_steps_per_task):
                # Parse observation for better decision making
                obs_data = parse_observation(obs)
                
                # Get task-specific strategy prompt
                prompt = get_strategy_prompt(task_name, obs_data)
                
                try:
                    response = llm_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3  # Lower temperature for consistent decisions
                    )
                    
                    qty_str = response.choices[0].message.content.strip()
                    qty = int(''.join(filter(str.isdigit, qty_str))) if any(c.isdigit() for c in qty_str) else 10
                    qty = max(1, min(qty, 50))  # Clamp between 1-50
                except Exception:
                    qty = 10  # Better fallback value
                
                action = {"t_id": step, "p_qty": qty}
                result = env_client.step(action)
                obs = result.observation
                reward = float(result.reward) if hasattr(result, 'reward') else 0.0
                
                total_reward += reward
                max_reward_seen = max(max_reward_seen, reward)
                steps_completed += 1
                
                # Print structured step output
                print(f"[STEP] step={step + 1} reward={reward:.2f}", flush=True)
                
                if hasattr(result, 'done') and result.done:
                    break
            
            # Better score calculation: average reward normalized to 0.0-1.0
            avg_reward = total_reward / max(steps_completed, 1) if steps_completed > 0 else 0.0
            final_score = min(1.0, max(0.0, avg_reward))  # Clamp to [0, 1]
            
            print(f"[END] task={task_name} score={final_score:.2f} steps={steps_completed}", flush=True)
                    
        except Exception as e:
            print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
            print(f"ERROR: {str(e)[:100]}", flush=True)

if __name__ == "__main__":
    run_agent()