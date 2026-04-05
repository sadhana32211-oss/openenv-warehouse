"""
Core Warehouse Environment implementing the OpenEnv standard API.

The environment simulates a warehouse where a robot must pick up packages
and deliver them to designated zones while avoiding obstacles.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from copy import deepcopy

from openenv_warehouse.config import WarehouseConfig
from openenv_warehouse.spaces import (
    DiscreteGrid, ActionSpace, BoxObservation, State
)


class WarehouseEnv:
    """
    Warehouse Robot Navigation Environment.
    
    This environment implements the standard OpenEnv API:
    - reset(): Resets the environment to an initial state
    - step(action): Takes an action and returns (observation, reward, terminated, truncated, info)
    - state(): Returns the current state dictionary
    
    The robot navigates a grid-based warehouse, picking up packages and
    delivering them to designated zones while avoiding obstacles.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi", None],
        "action_description": {
            0: "Move Up",
            1: "Move Down", 
            2: "Move Left",
            3: "Move Right",
            4: "Pick up package",
            5: "Deliver package",
            6: "No-op (wait)",
        }
    }
    
    def __init__(self, config: Optional[WarehouseConfig] = None):
        """
        Initialize the warehouse environment.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or WarehouseConfig()
        self.config.validate()
        
        # Initialize spaces
        self.grid = DiscreteGrid(self.config.grid_height, self.config.grid_width)
        self.action_space = ActionSpace()
        self.observation_space = BoxObservation(self.config)
        
        # Environment state (will be initialized in reset)
        self.robot_position: Tuple[int, int] = (0, 0)
        self.robot_inventory: List[int] = []
        self.obstacles: np.ndarray = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int8)
        self.packages: Dict[int, Dict[str, Any]] = {}
        self.delivery_zones: Dict[int, Dict[str, Any]] = {}
        
        # Tracking variables
        self.steps_taken: int = 0
        self.total_pickups: int = 0
        self.total_deliveries: int = 0
        self.total_collisions: int = 0
        
        # Episode state
        self._terminated: bool = False
        self._truncated: bool = False
        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional seed for reproducibility.
            options: Additional options for reset (e.g., specific layout).
            
        Returns:
            observation: Initial observation as numpy array.
            info: Additional information dictionary.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.config.seed = seed
        
        # Reset tracking variables
        self.steps_taken = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_collisions = 0
        self._terminated = False
        self._truncated = False
        self.robot_inventory = []
        
        # Generate warehouse layout
        self._generate_obstacles()
        self._place_robot()
        self._generate_packages()
        self._generate_delivery_zones()
        
        # Generate initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action from the action space (0-6).
            
        Returns:
            observation: New observation as numpy array.
            reward: Reward for the action.
            terminated: Whether the episode has terminated (task completed).
            truncated: Whether the episode has been truncated (time limit).
            info: Additional information dictionary.
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in [0, 6].")
        
        # Execute action and calculate reward
        reward = self.config.reward_step  # Small step penalty
        action_name = self.action_space.name(action)
        
        if action == 0:  # Move Up
            reward += self._move_robot(-1, 0)
        elif action == 1:  # Move Down
            reward += self._move_robot(1, 0)
        elif action == 2:  # Move Left
            reward += self._move_robot(0, -1)
        elif action == 3:  # Move Right
            reward += self._move_robot(0, 1)
        elif action == 4:  # Pick up package
            reward += self._pickup_package()
        elif action == 5:  # Deliver package
            reward += self._deliver_package()
        elif action == 6:  # No-op
            pass  # Just incur the step penalty
        
        # Increment step counter
        self.steps_taken += 1
        
        # Check termination conditions
        self._check_termination()
        
        # Generate observation and info
        observation = self._get_observation()
        info = self._get_info()
        info["action_name"] = action_name
        
        return observation, reward, self._terminated, self._truncated, info
    
    def state(self) -> State:
        """
        Get the complete current state of the environment.
        
        Returns:
            State object containing all environment information.
        """
        return State(
            robot_position=self.robot_position,
            robot_inventory=list(self.robot_inventory),
            robot_capacity=self.config.robot_capacity,
            packages=deepcopy(self.packages),
            delivery_zones=deepcopy(self.delivery_zones),
            obstacles=self.obstacles.copy(),
            grid_height=self.config.grid_height,
            grid_width=self.config.grid_width,
            steps_taken=self.steps_taken,
            max_steps=self.config.max_steps,
            total_pickups=self.total_pickups,
            total_deliveries=self.total_deliveries,
            total_collisions=self.total_collisions,
        )
    
    def render(self, mode: Optional[str] = None) -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Render mode ("human", "rgb_array", "ansi", or None).
            
        Returns:
            Rendered output depending on mode.
        """
        render_mode = mode or self.config.render_mode
        
        if render_mode == "ansi":
            return self._render_ansi()
        elif render_mode == "human":
            print(self._render_ansi())
            return None
        elif render_mode == "rgb_array":
            return self._render_rgb()
        else:
            return None
    
    def close(self):
        """Clean up the environment (if needed)."""
        pass
    
    # ==================== Internal Methods ====================
    
    def _generate_obstacles(self):
        """Generate random obstacles on the grid."""
        self.obstacles = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int8)
        
        num_obstacles = int(
            self.config.grid_height * self.config.grid_width * self.config.obstacle_density
        )
        
        placed = 0
        attempts = 0
        max_attempts = num_obstacles * 10
        
        while placed < num_obstacles and attempts < max_attempts:
            row = self._rng.integers(0, self.config.grid_height)
            col = self._rng.integers(0, self.config.grid_width)
            
            # Don't place obstacles in corners (for robot start and delivery zones)
            is_corner = (row < 2 and col < 2) or \
                       (row < 2 and col >= self.config.grid_width - 2) or \
                       (row >= self.config.grid_height - 2 and col < 2) or \
                       (row >= self.config.grid_height - 2 and col >= self.config.grid_width - 2)
            
            if self.obstacles[row, col] == 0 and not is_corner:
                self.obstacles[row, col] = 1
                placed += 1
            
            attempts += 1
    
    def _place_robot(self):
        """Place the robot in the top-left corner."""
        self.robot_position = (0, 0)
    
    def _generate_packages(self):
        """Generate packages at random locations."""
        self.packages = {}
        
        for i in range(self.config.num_packages):
            pos = self._find_empty_position()
            self.packages[i] = {
                "position": pos,
                "picked_up": False,
                "delivered": False,
            }
    
    def _generate_delivery_zones(self):
        """Generate delivery zones at random locations."""
        self.delivery_zones = {}
        
        for i in range(self.config.num_delivery_zones):
            pos = self._find_empty_position()
            self.delivery_zones[i] = {
                "position": pos,
                "packages_delivered": 0,
            }
    
    def _find_empty_position(self) -> Tuple[int, int]:
        """Find an empty position on the grid (not occupied by obstacle or other entities)."""
        occupied = set()
        occupied.add(self.robot_position)
        occupied.update(pkg["position"] for pkg in self.packages.values())
        occupied.update(zone["position"] for zone in self.delivery_zones.values())
        
        for _ in range(1000):
            row = self._rng.integers(0, self.config.grid_height)
            col = self._rng.integers(0, self.config.grid_width)
            pos = (row, col)
            
            if pos not in occupied and self.obstacles[row, col] == 0:
                return pos
        
        # Fallback: find any empty cell
        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                if (row, col) not in occupied and self.obstacles[row, col] == 0:
                    return (row, col)
        
        raise RuntimeError("No empty position available!")
    
    def _move_robot(self, drow: int, dcol: int) -> float:
        """
        Move the robot by the given delta.
        
        Returns:
            Reward for the movement.
        """
        new_row = self.robot_position[0] + drow
        new_col = self.robot_position[1] + dcol
        
        # Check bounds
        if not (0 <= new_row < self.config.grid_height and 
                0 <= new_col < self.config.grid_width):
            self.total_collisions += 1
            return self.config.reward_collision
        
        # Check obstacles
        if self.obstacles[new_row, new_col] == 1:
            self.total_collisions += 1
            return self.config.reward_collision
        
        # Valid move
        self.robot_position = (new_row, new_col)
        return 0.0
    
    def _pickup_package(self) -> float:
        """
        Attempt to pick up a package at the current position.
        
        Returns:
            Reward for the pickup attempt.
        """
        # Check if at capacity
        if len(self.robot_inventory) >= self.config.robot_capacity:
            return self.config.reward_invalid
        
        # Find package at current position
        for pkg_id, pkg in self.packages.items():
            if (pkg["position"] == self.robot_position and 
                not pkg["picked_up"] and 
                not pkg["delivered"]):
                
                pkg["picked_up"] = True
                self.robot_inventory.append(pkg_id)
                self.total_pickups += 1
                return self.config.reward_pickup
        
        return self.config.reward_invalid
    
    def _deliver_package(self) -> float:
        """
        Attempt to deliver a package at a delivery zone.
        
        Returns:
            Reward for the delivery attempt.
        """
        # Check if carrying any packages
        if not self.robot_inventory:
            return self.config.reward_invalid
        
        # Check if at a delivery zone
        for zone_id, zone in self.delivery_zones.items():
            if zone["position"] == self.robot_position:
                # Deliver one package
                pkg_id = self.robot_inventory.pop(0)
                self.packages[pkg_id]["delivered"] = True
                zone["packages_delivered"] += 1
                self.total_deliveries += 1
                return self.config.reward_delivery
        
        return self.config.reward_invalid
    
    def _check_termination(self):
        """Check if the episode should terminate or truncate."""
        # Check if all packages delivered
        all_delivered = all(pkg["delivered"] for pkg in self.packages.values())
        if all_delivered and self.config.terminate_on_complete:
            self._terminated = True
        
        # Check time limit
        if self.steps_taken >= self.config.max_steps:
            self._truncated = self.config.truncate_on_timeout
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the current observation as a normalized numpy array.
        
        Returns:
            Observation array.
        """
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        idx = 0
        
        # Robot position (normalized)
        obs[idx] = self.robot_position[0] / (self.config.grid_height - 1)
        obs[idx + 1] = self.robot_position[1] / (self.config.grid_width - 1)
        idx += 2
        
        # Inventory status (fill ratio)
        obs[idx] = len(self.robot_inventory) / self.config.robot_capacity
        idx += 1
        
        # Package positions and status
        for pkg in self.packages.values():
            pos = pkg["position"]
            obs[idx] = pos[0] / (self.config.grid_height - 1)
            obs[idx + 1] = pos[1] / (self.config.grid_width - 1)
            idx += 2
        
        # Package pickup status
        for pkg in self.packages.values():
            obs[idx] = 1.0 if pkg["picked_up"] else 0.0
            idx += 1
        
        # Delivery zone positions
        for zone in self.delivery_zones.values():
            pos = zone["position"]
            obs[idx] = pos[0] / (self.config.grid_height - 1)
            obs[idx + 1] = pos[1] / (self.config.grid_width - 1)
            idx += 2
        
        # Obstacle map
        obs[idx:] = self.obstacles.flatten().astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Generate additional information dictionary.
        
        Returns:
            Info dictionary.
        """
        return {
            "steps_remaining": self.config.max_steps - self.steps_taken,
            "packages_delivered": self.total_deliveries,
            "total_packages": self.config.num_packages,
            "inventory_count": len(self.robot_inventory),
            "inventory_capacity": self.config.robot_capacity,
            "collisions": self.total_collisions,
        }
    
    def _render_ansi(self) -> str:
        """
        Render the environment as ASCII text.
        
        Returns:
            ASCII representation of the warehouse.
        """
        # Create grid
        grid = [["." for _ in range(self.config.grid_width)] for _ in range(self.config.grid_height)]
        
        # Place obstacles
        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                if self.obstacles[row, col] == 1:
                    grid[row][col] = "#"
        
        # Place packages (not picked up)
        for pkg_id, pkg in self.packages.items():
            if not pkg["picked_up"]:
                row, col = pkg["position"]
                grid[row][col] = "P"
        
        # Place delivery zones
        for zone_id, zone in self.delivery_zones.items():
            row, col = zone["position"]
            delivered = zone["packages_delivered"]
            grid[row][col] = str(delivered) if delivered > 0 else "D"
        
        # Place robot
        robot_row, robot_col = self.robot_position
        grid[robot_row][robot_col] = "R"
        
        # Build output
        lines = []
        lines.append(f"Warehouse - Step: {self.steps_taken}/{self.config.max_steps}")
        lines.append(f"Inventory: {len(self.robot_inventory)}/{self.config.robot_capacity}")
        lines.append(f"Delivered: {self.total_deliveries}/{self.config.num_packages}")
        lines.append("")
        lines.append("  " + " ".join(str(i % 10) for i in range(self.config.grid_width)))
        
        for row_idx, row in enumerate(grid):
            lines.append(f"{row_idx} " + " ".join(row))
        
        return "\n".join(lines)
    
    def _render_rgb(self) -> np.ndarray:
        """
        Render the environment as an RGB array.
        
        Returns:
            RGB numpy array of the warehouse.
        """
        # Simple RGB rendering (can be enhanced with pygame/mpl)
        cell_size = 40
        img = np.zeros((
            self.config.grid_height * cell_size,
            self.config.grid_width * cell_size,
            3
        ), dtype=np.uint8)
        
        # Colors
        COLOR_BG = np.array([255, 255, 255], dtype=np.uint8)      # White
        COLOR_OBSTACLE = np.array([100, 100, 100], dtype=np.uint8)  # Gray
        COLOR_PACKAGE = np.array([255, 200, 0], dtype=np.uint8)     # Yellow
        COLOR_ZONE = np.array([0, 200, 100], dtype=np.uint8)        # Green
        COLOR_ROBOT = np.array([200, 0, 0], dtype=np.uint8)         # Red
        
        # Fill background
        img[:, :] = COLOR_BG
        
        # Draw obstacles
        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                if self.obstacles[row, col] == 1:
                    y1, y2 = row * cell_size, (row + 1) * cell_size
                    x1, x2 = col * cell_size, (col + 1) * cell_size
                    img[y1:y2, x1:x2] = COLOR_OBSTACLE
        
        # Draw packages
        for pkg in self.packages.values():
            if not pkg["picked_up"]:
                row, col = pkg["position"]
                y1, y2 = row * cell_size, (row + 1) * cell_size
                x1, x2 = col * cell_size, (col + 1) * cell_size
                img[y1:y2, x1:x2] = COLOR_PACKAGE
        
        # Draw delivery zones
        for zone in self.delivery_zones.values():
            row, col = zone["position"]
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            img[y1:y2, x1:x2] = COLOR_ZONE
        
        # Draw robot
        row, col = self.robot_position
        y1, y2 = row * cell_size, (row + 1) * cell_size
        x1, x2 = col * cell_size, (col + 1) * cell_size
        img[y1:y2, x1:x2] = COLOR_ROBOT
        
        return img