"""
Observation and Action space definitions for the Warehouse environment.

Follows OpenAI Gymnasium space conventions for compatibility.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field


class DiscreteGrid:
    """
    Represents a discrete grid space for the warehouse floor.
    
    Attributes:
        height: Grid height (rows)
        width: Grid width (columns)
    """
    
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.n = height * width
    
    def sample(self) -> Tuple[int, int]:
        """Sample a random grid position."""
        return (np.random.randint(self.height), np.random.randint(self.width))
    
    def contains(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid."""
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width
    
    def flatten(self, pos: Tuple[int, int]) -> int:
        """Convert 2D position to flat index."""
        return pos[0] * self.width + pos[1]
    
    def unflatten(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to 2D position."""
        return (idx // self.width, idx % self.width)
    
    def __repr__(self):
        return f"DiscreteGrid({self.height}, {self.width})"


class ActionSpace:
    """
    Discrete action space for robot movements and interactions.
    
    Actions:
        0: Move Up
        1: Move Down
        2: Move Left
        3: Move Right
        4: Pick up package (if at package location and not at capacity)
        5: Deliver package (if at delivery zone and carrying packages)
        6: No-op (wait)
    """
    
    ACTION_NAMES = ["up", "down", "left", "right", "pickup", "deliver", "noop"]
    NUM_ACTIONS = len(ACTION_NAMES)
    
    def __init__(self):
        self.n = self.NUM_ACTIONS
    
    def sample(self) -> int:
        """Sample a random action."""
        return np.random.randint(self.n)
    
    def contains(self, action: int) -> bool:
        """Check if an action is valid."""
        return 0 <= action < self.n
    
    def name(self, action: int) -> str:
        """Get the name of an action."""
        return self.ACTION_NAMES[action] if self.contains(action) else "unknown"
    
    def __repr__(self):
        return f"ActionSpace(n={self.n}, actions={self.ACTION_NAMES})"


class BoxObservation:
    """
    Box observation space for the warehouse environment.
    
    The observation is a flat vector containing:
    - Robot position (2 values: normalized row, col)
    - Robot inventory status (1 value: packages carried / capacity)
    - Package positions (num_packages * 2 values: normalized positions)
    - Package pickup status (num_packages values: 0=available, 1=picked up)
    - Delivery zone positions (num_zones * 2 values: normalized positions)
    - Obstacle map (flattened grid: 0=free, 1=obstacle)
    """
    
    def __init__(self, config):
        self.config = config
        
        # Calculate observation dimensions
        self.pos_dim = 2  # robot position (row, col)
        self.inventory_dim = 1  # inventory fill ratio
        self.package_pos_dim = config.num_packages * 2
        self.package_status_dim = config.num_packages
        self.zone_pos_dim = config.num_delivery_zones * 2
        self.obstacle_dim = config.grid_height * config.grid_width
        
        self.shape = (
            self.pos_dim + 
            self.inventory_dim + 
            self.package_pos_dim + 
            self.package_status_dim + 
            self.zone_pos_dim + 
            self.obstacle_dim,
        )
        
        self.low = np.zeros(self.shape, dtype=np.float32)
        self.high = np.ones(self.shape, dtype=np.float32)
    
    def sample(self) -> np.ndarray:
        """Sample a random observation."""
        return np.random.uniform(self.low, self.high).astype(np.float32)
    
    def contains(self, obs: np.ndarray) -> bool:
        """Check if an observation is valid."""
        return (obs.shape == self.shape and 
                np.all(obs >= self.low) and 
                np.all(obs <= self.high))
    
    def __repr__(self):
        return f"BoxObservation(shape={self.shape})"


@dataclass
class State:
    """
    Complete state representation of the warehouse environment.
    
    This is returned by the state() method and contains all information
    about the current environment state.
    """
    robot_position: Tuple[int, int]
    robot_inventory: List[int]  # List of package IDs being carried
    robot_capacity: int
    
    packages: Dict[int, Dict[str, Any]]  # package_id -> {position, picked_up, delivered}
    delivery_zones: Dict[int, Dict[str, Any]]  # zone_id -> {position, packages_delivered}
    
    obstacles: np.ndarray  # 2D binary grid
    grid_height: int
    grid_width: int
    
    steps_taken: int
    max_steps: int
    
    # Tracking metrics
    total_pickups: int
    total_deliveries: int
    total_collisions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        return {
            "robot": {
                "position": list(self.robot_position),
                "inventory": list(self.robot_inventory),
                "capacity": self.robot_capacity,
            },
            "packages": {
                str(k): {
                    "position": list(v["position"]),
                    "picked_up": v["picked_up"],
                    "delivered": v["delivered"],
                }
                for k, v in self.packages.items()
            },
            "delivery_zones": {
                str(k): {
                    "position": list(v["position"]),
                    "packages_delivered": v["packages_delivered"],
                }
                for k, v in self.delivery_zones.items()
            },
            "obstacles": self.obstacles.tolist(),
            "grid_size": [self.grid_height, self.grid_width],
            "steps": {
                "taken": self.steps_taken,
                "max": self.max_steps,
            },
            "metrics": {
                "total_pickups": self.total_pickups,
                "total_deliveries": self.total_deliveries,
                "total_collisions": self.total_collisions,
            },
        }