"""
Integration tests for full episode completion and task runnability.
"""

from server.environment import SCEnv
from envs.sc_env.models import SCAct


def test_full_episode_completeness():
    """Verify full episode runs without error"""
    env = SCEnv()
    obs = env.reset(seed=42, episode_id="supply_chain_medium-0")
    
    total_reward = 0.0
    step_count = 0
    
    for step in range(100):
        action = SCAct(supplier_id=1, purchase_qty=15)
        obs, reward, done = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        
        if done:
            break
    
    # Verify episode completed properly
    assert step_count == 50, f"Medium task should be 50 steps, got {step_count}"
    assert -50 < total_reward < 50, f"Total reward in reasonable range: {total_reward}"
    
    # Verify grading works
    grade = env.grade_episode()
    assert 0 < grade < 1, f"Grade out of range: {grade}"


def test_all_tasks_runnable():
    """Verify all three tasks can run to completion"""
    tasks = [
        "supply_chain_easy-0",
        "supply_chain_medium-0",
        "supply_chain_hard-0"
    ]
    
    for task_id in tasks:
        env = SCEnv()
        env.reset(seed=42, episode_id=task_id)
        
        step_count = 0
        for _ in range(50):
            action = SCAct(supplier_id=1, purchase_qty=15)
            obs, reward, done = env.step(action)
            step_count += 1
            if done:
                break
        
        grade = env.grade_episode()
        assert 0 < grade < 1, f"Task {task_id} failed grading: {grade}"
        assert step_count == 50, f"Task {task_id} didn't complete: {step_count} steps"


def test_industry_metrics_calculation():
    """Verify industry metrics are calculated correctly"""
    env = SCEnv()
    env.reset(seed=42, episode_id="supply_chain_easy-0")
    
    for _ in range(50):
        action = SCAct(supplier_id=1, purchase_qty=15)
        obs, reward, done = env.step(action)
        if done:
            break
    
    metrics = env.calculate_industry_metrics()
    
    # Verify all metrics are present
    assert "fill_rate" in metrics
    assert "perfect_order_rate" in metrics
    assert "inventory_turns" in metrics
    assert "days_supply" in metrics
    assert "cost_per_unit" in metrics
    
    # Verify metrics are in reasonable ranges
    assert 0 <= metrics["fill_rate"] <= 1.0
    assert 0 <= metrics["perfect_order_rate"] <= 1.0
    assert metrics["inventory_turns"] >= 0
    assert metrics["days_supply"] >= 0
    assert metrics["cost_per_unit"] >= 0


def test_multi_supplier_selection():
    """Verify agent can select different suppliers"""
    env = SCEnv()
    env.reset(seed=42, episode_id="supply_chain_medium-0")
    
    # Use each supplier
    suppliers_used = set()
    for step in range(50):
        supplier_id = step % 3  # Cycle through suppliers
        action = SCAct(supplier_id=supplier_id, purchase_qty=10)
        obs, reward, done = env.step(action)
        suppliers_used.add(supplier_id)
        if done:
            break
    
    assert len(suppliers_used) >= 2, "Should be able to use multiple suppliers"


def test_difficulty_scaling():
    """Verify difficulty metrics scale appropriately"""
    easy_env = SCEnv()
    easy_env.reset(seed=42, episode_id="supply_chain_easy-0")
    
    medium_env = SCEnv()
    medium_env.reset(seed=42, episode_id="supply_chain_medium-0")
    
    hard_env = SCEnv()
    hard_env.reset(seed=42, episode_id="supply_chain_hard-0")
    
    # Easy should have lower demand variance than hard
    assert easy_env.demand_variance < medium_env.demand_variance
    assert medium_env.demand_variance < hard_env.demand_variance
    
    # Easy should have larger capacity than hard
    assert easy_env.warehouse_capacity > medium_env.warehouse_capacity
    assert medium_env.warehouse_capacity > hard_env.warehouse_capacity
    
    # Hard should have higher shock probability
    assert easy_env.demand_shock_probability < hard_env.demand_shock_probability
