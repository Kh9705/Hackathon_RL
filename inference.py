import os
import time
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

def run_agent():
    print("--- Starting Compliant LLM-Driven Agent ---")
    try:
        result = env_client.reset()
        obs = result.observation
        
        for step in range(10): # Under 20-minute limit
            # Ask the LLM to process your specific variables
            prompt = (f"Observation: {obs}. You are a supply chain manager. "
                      f"If pending orders (p_ord) > 40, order 20. Otherwise order 5. "
                      f"Return ONLY the integer for the purchase quantity (p_qty).")
            
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                qty_str = response.choices[0].message.content.strip()
                qty = int(''.join(filter(str.isdigit, qty_str)))
            except Exception:
                qty = 5  # Fallback
            
            action = {"t_id": step, "p_qty": qty}
            result = env_client.step(action)
            obs = result.observation
            
            print(f"Step {step} | LLM Qty: {qty} | Reward: {result.reward}")
            if result.done: break
                
    except Exception as e:
        print(f"Inference Error: {e}")

if __name__ == "__main__":
    run_agent()