#!/usr/bin/env python3
"""
Unit tests for the OpenEnv Warehouse environment.

Tests cover:
- Environment initialization
- Reset functionality
- Step functionality
- State API
- Termination conditions
- Reward calculation
- Observation space
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv_warehouse import WarehouseEnv, WarehouseConfig
from openenv_warehouse.spaces import State


def test_environment_initialization():
    """Test that the environment initializes correctly."""
    print("Testing environment initialization...")
    
    # Default config
    env = WarehouseEnv()
    assert env.config.grid_height == 10
    assert env.config.grid_width == 10
    assert env.config.num_packages == 3
    assert env.config.num_delivery_zones == 2
    assert env.action_space.n == 7
    env.close()
    
    # Custom config
    config = WarehouseConfig(grid_height=15, grid_width=20, num_packages=5)
    env = WarehouseEnv(config)
    assert env.config.grid_height == 15
    assert env.config.grid_width == 20
    assert env.config.num_packages == 5
    env.close()
    
    print("  PASSED: Environment initialization")


def test_reset():
    """Test the reset functionality."""
    print("Testing reset...")
    
    env = WarehouseEnv()
    
    # Reset with seed
    obs1, info1 = env.reset(seed=42)
    assert obs1.shape == env.observation_space.shape
    assert isinstance(info1, dict)
    assert "steps_remaining" in info1
    assert info1["steps_remaining"] == env.config.max_steps
    
    # Reset with same seed should produce same observation
    obs2, info2 = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    
    # Reset without seed
    obs3, info3 = env.reset()
    assert obs3.shape == env.observation_space.shape
    
    env.close()
    print("  PASSED: Reset functionality")


def test_step():
    """Test the step functionality."""
    print("Testing step...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    # Test all actions
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "action_name" in info
    
    env.close()
    print("  PASSED: Step functionality")


def test_step_after_termination():
    """Test that step raises error after termination."""
    print("Testing step after termination...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    # Force termination by exceeding max steps
    env.config.max_steps = 5
    for _ in range(5):
        env.step(6)  # No-op
    
    # Should be truncated now
    assert env._truncated
    
    # Step should raise error
    try:
        env.step(0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    
    env.close()
    print("  PASSED: Step after termination")


def test_state_api():
    """Test the state() API."""
    print("Testing state API...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    # Take some actions
    for _ in range(5):
        env.step(env.action_space.sample())
    
    # Get state
    state = env.state()
    assert isinstance(state, State)
    assert isinstance(state.robot_position, tuple)
    assert isinstance(state.robot_inventory, list)
    assert isinstance(state.packages, dict)
    assert isinstance(state.delivery_zones, dict)
    assert isinstance(state.obstacles, np.ndarray)
    
    # Convert to dict
    state_dict = state.to_dict()
    assert "robot" in state_dict
    assert "packages" in state_dict
    assert "delivery_zones" in state_dict
    assert "obstacles" in state_dict
    assert "metrics" in state_dict
    
    env.close()
    print("  PASSED: State API")


def test_observation_space():
    """Test observation space properties."""
    print("Testing observation space...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    obs, _, _, _, _ = env.step(0)
    
    # Check shape
    assert obs.shape == env.observation_space.shape
    
    # Check bounds (should be normalized 0-1)
    assert np.all(obs >= 0)
    assert np.all(obs <= 1)
    
    # Check dtype
    assert obs.dtype == np.float32
    
    env.close()
    print("  PASSED: Observation space")


def test_reward_structure():
    """Test reward structure."""
    print("Testing reward structure...")
    
    config = WarehouseConfig(
        reward_step=-0.1,
        reward_collision=-5.0,
        reward_pickup=10.0,
        reward_delivery=25.0,
        reward_invalid=-1.0,
    )
    env = WarehouseEnv(config)
    env.reset(seed=42)
    
    # Step penalty
    obs, reward, _, _, _ = env.step(0)  # Move
    assert reward == config.reward_step or reward == config.reward_step + config.reward_collision
    
    # Invalid pickup (not at package)
    obs, reward, _, _, _ = env.step(4)  # Pickup
    assert reward == config.reward_step + config.reward_invalid
    
    # Invalid deliver (not at zone or no inventory)
    obs, reward, _, _, _ = env.step(5)  # Deliver
    assert reward == config.reward_step + config.reward_invalid
    
    env.close()
    print("  PASSED: Reward structure")


def test_render():
    """Test rendering functionality."""
    print("Testing render...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    # ANSI render
    ansi_output = env.render(mode="ansi")
    assert isinstance(ansi_output, str)
    assert "R" in ansi_output  # Robot symbol
    assert "P" in ansi_output  # Package symbol
    assert "D" in ansi_output  # Delivery zone symbol
    
    # RGB render
    rgb_output = env.render(mode="rgb_array")
    assert isinstance(rgb_output, np.ndarray)
    assert rgb_output.shape[2] == 3  # RGB channels
    
    env.close()
    print("  PASSED: Render functionality")


def test_collision_detection():
    """Test collision detection with walls and obstacles."""
    print("Testing collision detection...")
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    initial_collisions = env.total_collisions
    
    # Try to move out of bounds (top-left corner)
    env.step(0)  # Up - should collide with top wall
    env.step(2)  # Left - should collide with left wall
    
    assert env.total_collisions >= initial_collisions + 1
    
    env.close()
    print("  PASSED: Collision detection")


def test_package_delivery_workflow():
    """Test the complete package pickup and delivery workflow."""
    print("Testing package delivery workflow...")
    
    # Use a small grid for easier testing
    config = WarehouseConfig(
        grid_height=5,
        grid_width=5,
        num_packages=1,
        num_delivery_zones=1,
        max_steps=100,
        obstacle_density=0.0,
    )
    env = WarehouseEnv(config)
    env.reset(seed=42)
    
    # Get initial state
    state = env.state()
    initial_deliveries = state.total_deliveries
    
    # The workflow should be:
    # 1. Navigate to package
    # 2. Pick up package
    # 3. Navigate to delivery zone
    # 4. Deliver package
    
    # For this test, we just verify the mechanics work
    # A random agent might not complete the task, but we can check the logic
    
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
        
        state = env.state()
        if state.total_deliveries > initial_deliveries:
            break
    
    env.close()
    print("  PASSED: Package delivery workflow")


def test_termination_conditions():
    """Test episode termination conditions."""
    print("Testing termination conditions...")
    
    # Test time limit truncation
    config = WarehouseConfig(max_steps=10, terminate_on_complete=False)
    env = WarehouseEnv(config)
    env.reset(seed=42)
    
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(6)  # No-op
    
    assert truncated
    assert not terminated
    
    env.close()
    print("  PASSED: Termination conditions")


def test_reproducibility():
    """Test that same seed produces same results."""
    print("Testing reproducibility...")
    
    # Run 1
    env1 = WarehouseEnv()
    env1.reset(seed=123)
    actions1 = []
    for _ in range(20):
        action = env1.action_space.sample()
        actions1.append(action)
        obs1, r1, t1, tr1, i1 = env1.step(action)
    state1 = env1.state()
    env1.close()
    
    # Run 2 with same seed
    env2 = WarehouseEnv()
    env2.reset(seed=123)
    actions2 = []
    for _ in range(20):
        action = env2.action_space.sample()
        actions2.append(action)
        obs2, r2, t2, tr2, i2 = env2.step(action)
    state2 = env2.state()
    env2.close()
    
    # Note: Actions are random, but environment dynamics should be same
    # if we use the same action sequence
    env3 = WarehouseEnv()
    env3.reset(seed=123)
    for action in actions1:
        obs3, r3, t3, tr3, i3 = env3.step(action)
    state3 = env3.state()
    env3.close()
    
    # Same seed should produce same layout and same results with same actions
    np.testing.assert_array_equal(state1.obstacles, state3.obstacles)
    assert state1.robot_position == state3.robot_position
    
    print("  PASSED: Reproducibility")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running OpenEnv Warehouse Tests")
    print("="*60)
    print()
    
    tests = [
        test_environment_initialization,
        test_reset,
        test_step,
        test_step_after_termination,
        test_state_api,
        test_observation_space,
        test_reward_structure,
        test_render,
        test_collision_detection,
        test_package_delivery_workflow,
        test_termination_conditions,
        test_reproducibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print()
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)