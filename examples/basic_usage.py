#!/usr/bin/env python3
"""
Basic usage example for the OpenEnv Warehouse environment.

This script demonstrates how to use the standard OpenEnv API:
- reset() to initialize the environment
- step(action) to take actions
- state() to get the full state
- render() to visualize the environment
"""

import numpy as np
from openenv_warehouse import WarehouseEnv, WarehouseConfig


def run_random_agent(num_episodes: int = 3, verbose: bool = True):
    """
    Run a random agent in the warehouse environment.
    
    Args:
        num_episodes: Number of episodes to run.
        verbose: Whether to print detailed information.
    """
    # Create environment with default config
    env = WarehouseEnv()
    
    for episode in range(num_episodes):
        # Reset environment
        observation, info = env.reset(seed=42 + episode)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            env.render()
            print(f"\nInitial observation shape: {observation.shape}")
            print(f"Observation sample: {observation[:10]}...")
        
        total_reward = 0.0
        step = 0
        
        while True:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if verbose and step <= 10:
                print(f"\nStep {step}:")
                print(f"  Action: {info.get('action_name', 'unknown')} ({action})")
                print(f"  Reward: {reward:.2f}")
                print(f"  Delivered: {info['packages_delivered']}/{info['total_packages']}")
                print(f"  Inventory: {info['inventory_count']}/{info['inventory_capacity']}")
            
            # Check if episode ended
            if terminated or truncated:
                if verbose:
                    print(f"\nEpisode ended after {step} steps!")
                    print(f"  Termination reason: {'Task completed' if terminated else 'Time limit'}")
                    print(f"  Total reward: {total_reward:.2f}")
                    print(f"  Packages delivered: {info['packages_delivered']}")
                    print(f"  Collisions: {info['collisions']}")
                break
        
        # Get full state
        state = env.state()
        if verbose:
            print(f"\nFinal state (dict format):")
            state_dict = state.to_dict()
            print(f"  Robot position: {state_dict['robot']['position']}")
            print(f"  Packages delivered: {state_dict['metrics']['total_deliveries']}")
    
    env.close()
    print(f"\nCompleted {num_episodes} episodes!")


def demonstrate_custom_config():
    """Demonstrate using a custom configuration."""
    print("\n" + "="*60)
    print("Custom Configuration Example")
    print("="*60)
    
    # Create custom config
    config = WarehouseConfig(
        grid_height=15,
        grid_width=15,
        num_packages=5,
        num_delivery_zones=3,
        max_steps=300,
        robot_capacity=2,
        obstacle_density=0.2,
        seed=123,
    )
    
    # Create environment with custom config
    env = WarehouseEnv(config)
    observation, info = env.reset()
    
    print(f"Grid size: {config.grid_height}x{config.grid_width}")
    print(f"Packages: {config.num_packages}")
    print(f"Delivery zones: {config.num_delivery_zones}")
    print(f"Max steps: {config.max_steps}")
    print(f"Robot capacity: {config.robot_capacity}")
    print(f"Obstacle density: {config.obstacle_density}")
    
    env.render()
    env.close()


def demonstrate_state_api():
    """Demonstrate the state() API."""
    print("\n" + "="*60)
    print("State API Example")
    print("="*60)
    
    env = WarehouseEnv()
    env.reset(seed=42)
    
    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)
    
    # Get full state
    state = env.state()
    
    print("\nState object attributes:")
    print(f"  Robot position: {state.robot_position}")
    print(f"  Robot inventory: {state.robot_inventory}")
    print(f"  Steps taken: {state.steps_taken}")
    print(f"  Total pickups: {state.total_pickups}")
    print(f"  Total deliveries: {state.total_deliveries}")
    print(f"  Total collisions: {state.total_collisions}")
    print(f"  Grid size: {state.grid_height}x{state.grid_width}")
    print(f"  Number of packages: {len(state.packages)}")
    print(f"  Number of delivery zones: {len(state.delivery_zones)}")
    
    # Convert to dict
    state_dict = state.to_dict()
    print(f"\nState as dictionary keys: {list(state_dict.keys())}")
    
    env.close()


if __name__ == "__main__":
    print("OpenEnv Warehouse - Basic Usage Examples")
    print("="*60)
    
    # Run random agent
    run_random_agent(num_episodes=2, verbose=True)
    
    # Custom config demo
    demonstrate_custom_config()
    
    # State API demo
    demonstrate_state_api()