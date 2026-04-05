"""
OpenEnv Warehouse - A real-world warehouse robot navigation environment.

This package provides a reinforcement learning environment where AI agents
control robots in a warehouse to pick up and deliver packages efficiently.

Standard OpenEnv API:
    - reset(): Resets the environment to an initial state
    - step(action): Takes an action and returns (observation, reward, terminated, truncated, info)
    - state(): Returns the current state dictionary
"""

from openenv_warehouse.environment import WarehouseEnv
from openenv_warehouse.spaces import DiscreteGrid, BoxObservation, ActionSpace, State
from openenv_warehouse.config import WarehouseConfig

__version__ = "1.0.0"
__all__ = ["WarehouseEnv", "WarehouseConfig", "DiscreteGrid", "BoxObservation", "ActionSpace", "State"]