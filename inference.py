# Updated inference.py
from openenv.core import SyncEnvClient, GenericEnvClient
import os
import time

# Use environment variables for API keys/URLs for security
API_KEY = os.getenv("OPENENV_API_KEY", "default_key")
BASE_URL = os.getenv("OPENENV_URL", "http://localhost:8000")

base_client = GenericEnvClient(BASE_URL)
client = SyncEnvClient(base_client)

def refined_agent():
    print("--- Starting Refined Supply Chain Agent ---")
    try:
        result = client.reset()
        obs = result.observation
        
        for step in range(10):
            # Logic: If pending orders are > 40, order 20 units. Otherwise, order 5.
            pending = obs.get('p_ord', 0)
            qty = 20 if pending > 40 else 5
            
            action = {"t_id": step, "p_qty": qty}
            result = client.step(action)
            
            obs = result.observation
            print(f"Step {step} | Pending: {pending} | Action: Order {qty} | Reward: {result.reward}")
            time.sleep(1)
            
            if result.done:
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    refined_agent()