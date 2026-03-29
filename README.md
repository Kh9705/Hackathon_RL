# 📦 Supply Chain Agentic AI - OpenEnv Hackathon 2026

An autonomous Reinforcement Learning (RL) agent designed to optimize inventory management and procurement strategies using the **OpenEnv** framework.

## 🚀 Overview
This project implements a custom supply chain environment where an agent must balance **Total Capacity (t_cap)** against **Pending Orders (p_ord)**. The goal is to maximize efficiency by dynamically adjusting **Purchase Quantity (p_qty)** without exceeding warehouse limits.

## 🏗️ Technical Architecture
- **Framework**: Built on the latest `openenv-core` 2026 standards.
- **Environment**: Custom Python-based RL environment (`SCEnv`) hosted via FastAPI and Uvicorn.
- **Deployment**: Containerized using **Docker** and deployed on **Hugging Face Spaces**.
- **Communication**: Synchronous client-server architecture using Pydantic models for data integrity.

## 📁 Project Structure
- `envs/sc_env/`: Core environment logic and state management.
- `envs/sc_env/models.py`: Pydantic schemas for Actions (`SCAct`) and Observations (`SCObs`).
- `envs/sc_env/server/app.py`: FastAPI entry point for the environment server.
- `inference.py`: The "Brain" – refined agent logic that uses threshold-based ordering to maximize rewards.
- `Dockerfile`: Optimized multi-stage build for cloud deployment.
- `openai.yaml`: OpenEnv manifest for environment discovery.

## 🧠 Agent Logic (The "Refined" Approach)
Unlike a static agent, this implementation uses a **dynamic threshold strategy**:
- **High Demand**: If `p_ord` > 40, the agent aggressively orders 20 units to prevent stockouts.
- **Maintenance**: If `p_ord` is stable, it orders 5 units to minimize holding costs.
- **Objective**: Maximize the cumulative reward by keeping the supply chain "fluid."

## 🛠️ How to Run Locally
1. Build the environment:
   `docker build -t sc_env -f Dockerfile .`
2. Run the server:
   `docker run -p 8000:8000 sc_env`
3. Execute the agent:
   `PYTHONPATH=. python inference.py`