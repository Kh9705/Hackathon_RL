"""
Unit tests for supply chain environment determinism and grading quality.
"""

import pytest
from server.environment import SCEnv
from envs.sc_env.models import SCAct


class TestEnvironmentDeterminism:
    """Verify environment is deterministic with seed control."""
    
    def test_reset_determinism(self):
        """Same seed → same initial state"""
        env1 = SCEnv()
        obs1 = env1.reset(seed=42, episode_id="supply_chain_easy-0")
        
        env2 = SCEnv()
        obs2 = env2.reset(seed=42, episode_id="supply_chain_easy-0")
        
        assert obs1.pending_orders == obs2.pending_orders
        assert obs1.warehouse_inventory == obs2.warehouse_inventory
        assert obs1.demand_rate == obs2.demand_rate

    def test_step_determinism(self):
        """Same seed + same action → same next state"""
        env1 = SCEnv()
        env1.reset(seed=42, episode_id="supply_chain_easy-0")
        action = SCAct(supplier_id=1, purchase_qty=10)
        obs1, reward1, done1 = env1.step(action)
        
        env2 = SCEnv()
        env2.reset(seed=42, episode_id="supply_chain_easy-0")
        obs2, reward2, done2 = env2.step(action)
        
        assert obs1.pending_orders == obs2.pending_orders
        assert reward1 == reward2
        assert done1 == done2

    def test_full_trajectory_determinism(self):
        """Full episode with seed is completely reproducible"""
        env1 = SCEnv()
        env1.reset(seed=123, episode_id="supply_chain_medium-0")
        rewards1 = []
        
        for _ in range(50):
            action = SCAct(supplier_id=1, purchase_qty=15)
            obs, reward, done = env1.step(action)
            rewards1.append(reward)
            if done:
                break
        
        env2 = SCEnv()
        env2.reset(seed=123, episode_id="supply_chain_medium-0")
        rewards2 = []
        
        for _ in range(50):
            action = SCAct(supplier_id=1, purchase_qty=15)
            obs, reward, done = env2.step(action)
            rewards2.append(reward)
            if done:
                break
        
        assert rewards1 == rewards2, "Full trajectory not deterministic"


class TestRewardBounds:
    """Verify rewards are always in valid range."""
    
    def test_reward_in_bounds(self):
        """All step rewards in [-1, 1]"""
        env = SCEnv()
        env.reset(seed=42, episode_id="supply_chain_hard-0")
        
        for step in range(100):
            action = SCAct(supplier_id=1, purchase_qty=10)
            obs, reward, done = env.step(action)
            
            assert isinstance(reward, float), f"Reward not float: {type(reward)}"
            assert -1.0 <= reward <= 1.0, f"Reward out of bounds at step {step}: {reward}"
            if done:
                break


class TestGraderQuality:
    """Verify graders produce sensible scores."""
    
    def test_easy_task_grades_well(self):
        """Easy task should be learnable - baseline agent gets decent score"""
        env = SCEnv()
        env.reset(seed=42, episode_id="supply_chain_easy-0")
        
        # Run simple policy
        for _ in range(50):
            action = SCAct(supplier_id=1, purchase_qty=15)
            obs, reward, done = env.step(action)
            if done:
                break
        
        grade = env.grade_episode()
        assert 0.4 < grade < 0.9, f"Easy task grade unreasonable: {grade}"

    def test_grading_consistency(self):
        """Grader deterministic for same trajectory"""
        env1 = SCEnv()
        env1.reset(seed=42, episode_id="supply_chain_medium-0")
        
        env2 = SCEnv()
        env2.reset(seed=42, episode_id="supply_chain_medium-0")
        
        action = SCAct(supplier_id=1, purchase_qty=15)
        for _ in range(50):
            env1.step(action)
            env2.step(action)
        
        grade1 = env1.grade_episode()
        grade2 = env2.grade_episode()
        
        assert grade1 == grade2, f"Grades inconsistent: {grade1} vs {grade2}"


class TestDifficultyProgression:
    """Verify tasks have proper difficulty progression."""
    
    def test_easy_easier_than_hard(self):
        """Easy task should score higher than hard with same policy"""
        # Run on easy
        env_easy = SCEnv()
        env_easy.reset(seed=42, episode_id="supply_chain_easy-0")
        
        # Run on hard
        env_hard = SCEnv()
        env_hard.reset(seed=42, episode_id="supply_chain_hard-0")
        
        action = SCAct(supplier_id=1, purchase_qty=15)
        for _ in range(50):
            env_easy.step(action)
            env_hard.step(action)
        
        grade_easy = env_easy.grade_episode()
        grade_hard = env_hard.grade_episode()
        
        assert grade_easy > grade_hard, f"Easy not easier: easy={grade_easy}, hard={grade_hard}"


class TestObservationTypes:
    """Verify observations are properly typed."""
    
    def test_observation_has_all_fields(self):
        """Observation contains all required fields"""
        env = SCEnv()
        obs = env.reset()
        
        assert hasattr(obs, 'pending_orders')
        assert hasattr(obs, 'warehouse_inventory')
        assert hasattr(obs, 'warehouse_capacity')
        assert hasattr(obs, 'demand_rate')
        assert hasattr(obs, 'info')

    def test_observation_types_correct(self):
        """All observation fields have correct types"""
        env = SCEnv()
        obs = env.reset(seed=42, episode_id="supply_chain_easy-0")
        
        assert isinstance(obs.pending_orders, int)
        assert isinstance(obs.warehouse_inventory, int)
        assert isinstance(obs.warehouse_capacity, int)
        assert isinstance(obs.demand_rate, float)
        assert isinstance(obs.info, dict)
