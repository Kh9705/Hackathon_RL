# 📦 Supply Chain Optimization - OpenEnv Hackathon 2026

## 🎯 Overview

This project implements a **realistic supply chain inventory management environment** using the OpenEnv framework. An AI agent learns to balance multiple conflicting objectives:

- **Demand fulfillment** - Meet customer orders (maximize revenue)
- **Inventory costs** - Minimize holding costs (minimize waste)
- **Shortage penalties** - Avoid stockouts (maintain reputation)

The environment features **stochastic demand**, **capacity constraints**, and **task-specific difficulty levels** that increase agent complexity.

---

## 🏗️ Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **Environment** (`server/environment.py`) | Simulates supply chain dynamics with realistic physics |
| **Models** (`shared/models.py`) | Pydantic data classes for type safety |
| **Server** (`server/app.py`) | FastAPI + Uvicorn for OpenEnv compliance |
| **Agent** (`inference.py`) | LLM-powered decision maker with fallback heuristics |
| **Config** (`openenv.yaml`) | Task definitions and grading criteria |

### Technology Stack

- **Framework**: OpenEnv Core 0.2.0+ (RL environment standard)
- **API**: FastAPI + Uvicorn (async HTTP server)
- **Models**: Pydantic (data validation)
- **LLM**: OpenAI Client (Claude, GPT-4, or any compatible endpoint)
- **Deployment**: Docker + Hugging Face Spaces

---

## 🎮 Environment Specification

### Action Space

The agent can perform one action per step:

```python
@dataclass
class SCAct:
    supplier_id: int      # Which supplier (0 = default, extensible)
    purchase_qty: int     # Units to order (0-warehouse_capacity)
```

**Valid range**: `purchase_qty ∈ [0, warehouse_capacity]`

### Observation Space

After each step, the agent receives:

```python
@dataclass
class SCObs:
    pending_orders: int        # Unfulfilled customer orders
    warehouse_inventory: int   # Current stock level
    warehouse_capacity: int    # Maximum storage capacity
    demand_rate: float         # Expected orders/step (forecast)
    info: Dict[str, Any]       # Additional metadata
```

### Reward Function

```
reward = fulfillment_rate - (normalized_costs / 100)

where:
  fulfillment_rate = fulfilled_orders / total_demand
  costs = holding_cost + stockout_penalty + order_cost
```

**Properties**:
- ✅ Bounded to [-1.0, 1.0] per step
- ✅ Positive for good decisions, negative for poor ones
- ✅ Normalized and interpretable

---

## 📊 Tasks (3 Difficulty Levels)

### Task 1: Easy - "Baseline Fulfillment"

**Scenario**: Low, predictable demand with forgiving constraints.

| Parameter | Value |
|-----------|-------|
| Base Demand | 8.0 orders/step |
| Demand Volatility | ±2.0 (low variance) |
| Warehouse Capacity | 150 units |
| Holding Cost | $0.30/unit/step |
| Stockout Penalty | $5.00/unmet order |
| Max Steps | 50 |

**Strategy**: Learn basic ordering rules without extreme complexity.

---

### Task 2: Medium - "Peak Season Stress"

**Scenario**: Moderate, variable demand with balanced constraints.

| Parameter | Value |
|-----------|-------|
| Base Demand | 15.0 orders/step |
| Demand Volatility | ±5.0 (moderate variance) |
| Warehouse Capacity | 100 units |
| Holding Cost | $0.50/unit/step |
| Stockout Penalty | $10.00/unmet order |
| Max Steps | 50 |

**Strategy**: Balance cost minimization with demand satisfaction.

---

### Task 3: Hard - "Resource Bottleneck"

**Scenario**: High, volatile demand with tight capacity constraints.

| Parameter | Value |
|-----------|-------|
| Base Demand | 25.0 orders/step |
| Demand Volatility | ±10.0 (high variance) |
| Warehouse Capacity | 80 units |
| Holding Cost | $0.80/unit/step |
| Stockout Penalty | $15.00/unmet order |
| Max Steps | 50 |

**Strategy**: Optimize under extreme pressure; capacity is the limiting factor.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (for deployment)
- pip or uv (package manager)

### Local Installation

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd sc-env
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"  # Or your LLM endpoint
   export MODEL_NAME="gpt-4"                         # Model to use
   export HF_TOKEN="hf_xxxxxxxxxxxx"                 # API key
   ```

3. **Run the server**:
   ```bash
   python -m server.app
   # Or: uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

4. **Run the agent** (in another terminal):
   ```bash
   python inference.py
   ```

### Docker Deployment

1. **Build**:
   ```bash
   docker build -t sc-env .
   ```

2. **Run**:
   ```bash
   docker run \
     -p 8000:8000 \
     -e API_BASE_URL \
     -e MODEL_NAME \
     -e HF_TOKEN\
     sc-env
   ```

---

## 🧠 Agent Strategy

The inference script uses a **hybrid approach**:

1. **LLM-Powered Decisions**: Sends task-specific prompts to Claude/GPT-4
2. **Heuristic Fallback**: Uses rule-based logic if LLM unavailable
3. **Difficulty Adaptation**: Different strategies for easy/medium/hard

### Example Agent Logic

```python
# For EASY task:
if pending_orders > 20:
    order = demand_rate * 1.5  # Aggressive
else:
    order = demand_rate * 0.8  # Conservative

# For HARD task:
order = max(pending_orders - inventory + buffer, 5)
order = min(order, capacity * 0.5)  # Respect capacity limit
```

---

## 📈 Expected Performance

With the provided agent:

| Task | Difficulty | Expected Score |
|------|-----------|-----------------|
| Easy | ⭐ | 0.70 - 0.85 |
| Medium | ⭐⭐ | 0.55 - 0.70 |
| Hard | ⭐⭐⭐ | 0.40 - 0.60 |

**Note**: Scores depend on random seed, LLM availability, and prompt quality.

---

## 📋 Project Files

```
sc-env/
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI server entry point
│   └── environment.py         # SCEnv simulation logic
├── shared/
│   ├── __init__.py
│   └── models.py              # Pydantic data classes
├── inference.py               # Agent logic (REQUIRED by spec)
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package metadata
├── openenv.yaml              # Task & environment config
└── README.md                 # This file
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Example |
|----------|----------|---------|
| `API_BASE_URL` | ✅ | `https://api.openai.com/v1` |
| `MODEL_NAME` | ✅ | `gpt-4` or `claude-3-sonnet` |
| `HF_TOKEN` | ✅ | `hf_xxxxxxxxxxx` |
| `OPENENV_URL` | ❌ | `http://localhost:8000` (default) |

### openenv.yaml

Defines tasks, action/observation spaces, and grading:

```yaml
version: "0.2.0"
name: "SC-Env-Supply-Chain"

action_space:
  supplier_id: "int"
  purchase_qty: "int"

observation_space:
  pending_orders: "int"
  warehouse_inventory: "int"
  warehouse_capacity: "int"
  demand_rate: "float"

tasks:
  - id: "supply_chain_easy"
    difficulty: "easy"
  - id: "supply_chain_medium"
    difficulty: "medium"
  - id: "supply_chain_hard"
    difficulty: "hard"
```

---

## 🧪 Testing

### Unit Test Environment

```python
from server.environment import SCEnv
from shared.models import SCAct

env = SCEnv()
obs = env.reset(seed=42, episode_id="supply_chain_medium-0")

for step in range(10):
    action = SCAct(supplier_id=0, purchase_qty=20)
    obs, reward, done = env.step(action)
    print(f"Step {step}: reward={reward:.2f}, inventory={obs.warehouse_inventory}")
    if done:
        break
```

### Validation Script

The hackathon provides a `validate.py` script that checks:
- ✅ Imports and syntax
- ✅ Dockerfile builds
- ✅ Server responds to `/reset` and `/step`
- ✅ Inference runs without errors
- ✅ Log format is correct
- ✅ Scores are in [0.0, 1.0]

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'server'"
**Solution**: Run from project root: `python -m inference.py`

### "[ERROR] Missing environment variables"
**Solution**: Set all three before running:
```bash
export API_BASE_URL="..."
export MODEL_NAME="..."
export HF_TOKEN="..."
```

### "Connection refused" (Docker)
**Solution**: Ensure server is running:
```bash
docker logs <container-id>  # Check logs
docker port <container-id>  # Verify port mapping
```

### Low scores on all tasks
**Solution**: Check:
1. Is the environment.step() returning dynamic rewards?
2. Are all 3 tasks producing different behavior?
3. Is the agent receiving observations correctly?

---

## 📚 Additional Resources

- **OpenEnv Docs**: https://docs.openenv.dev
- **OpenAI API**: https://platform.openai.com/docs
- **FastAPI**: https://fastapi.tiangolo.com
- **Pydantic**: https://docs.pydantic.dev



## 📄 License

MIT License - See LICENSE file for details

---

