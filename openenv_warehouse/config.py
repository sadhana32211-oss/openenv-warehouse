"""
Configuration settings for the Warehouse environment.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class WarehouseConfig:
    """Configuration parameters for the warehouse environment."""
    
    # Grid dimensions
    grid_height: int = 10
    grid_width: int = 10
    
    # Number of packages and delivery zones
    num_packages: int = 3
    num_delivery_zones: int = 2
    
    # Time limits
    max_steps: int = 200
    
    # Robot parameters
    robot_capacity: int = 3  # Max packages robot can carry
    robot_speed: int = 1     # Cells per step
    
    # Reward values
    reward_pickup: float = 10.0      # Reward for picking up a package
    reward_delivery: float = 25.0    # Reward for delivering a package
    reward_step: float = -0.1        # Small negative reward per step (encourage efficiency)
    reward_collision: float = -5.0   # Penalty for hitting obstacles/walls
    reward_invalid: float = -1.0     # Penalty for invalid actions
    
    # Termination conditions
    terminate_on_complete: bool = True   # End episode when all packages delivered
    truncate_on_timeout: bool = True     # Truncate episode on max_steps
    
    # Obstacle density (percentage of grid cells that are obstacles)
    obstacle_density: float = 0.15
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # Visualization
    render_mode: str = "human"  # "human", "rgb_array", "ansi", None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.grid_height < 5 or self.grid_width < 5:
            raise ValueError("Grid dimensions must be at least 5x5")
        if self.obstacle_density < 0 or self.obstacle_density > 0.5:
            raise ValueError("Obstacle density must be between 0 and 0.5")
        if self.num_packages < 1:
            raise ValueError("Must have at least 1 package")
        if self.num_delivery_zones < 1:
            raise ValueError("Must have at least 1 delivery zone")
        if self.robot_capacity < 1:
            raise ValueError("Robot capacity must be at least 1")
        return True
    
    @staticmethod
    def create_default() -> 'WarehouseConfig':
        """Create a default configuration."""
        return WarehouseConfig()