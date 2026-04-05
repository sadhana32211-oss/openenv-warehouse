# OpenEnv Warehouse

A real-world reinforcement learning environment for warehouse robot navigation, implementing the standard OpenEnv API (`reset()`, `step()`, `state()`).

## Overview

The **OpenEnv Warehouse** environment simulates a warehouse where an AI agent controls a robot that must:
- Navigate a grid-based warehouse floor
- Pick up packages from various locations
- Deliver packages to designated delivery zones
- Avoid obstacles (shelving, walls, etc.)
- Maximize efficiency (minimize steps and collisions)

This environment is designed to be a practical, real-world problem that tests an AI agent's ability to plan, navigate, and make decisions in a constrained environment.

## Installation

```bash
# Clone or download the project
cd openenv_project

# Install dependencies (if any external packages needed)
pip install numpy

# That's it! No other dependencies required.
```

## Quick Start

```python
from openenv_warehouse import WarehouseEnv

# Create environment
env = WarehouseEnv()

# Reset to initial state
observation, info = env.reset(seed=42)

# Take actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

# Get full state
state = env.state()
print(f"Packages delivered: {state.total_deliveries}")

env.close()
```

## Standard OpenEnv API

### `reset(seed=None, options=None)`
Resets the environment to an initial state and returns:
- `observation`: Initial observation as a numpy array
- `info`: Additional information dictionary

### `step(action)`
Takes an action and returns:
- `observation`: New observation as a numpy array
- `reward`: Float reward for the action
- `terminated`: Boolean indicating if the task is complete
- `truncated`: Boolean indicating if the time limit was reached
- `info`: Additional information dictionary

### `state()`
Returns a `State` object containing the complete environment state, including:
- Robot position and inventory
- Package positions and status
- Delivery zone positions
- Obstacle map
- Tracking metrics

### `render(mode="human")`
Renders the environment in various modes:
- `"human"`: Print to console
- `"ansi"`: Return ASCII string
- `"rgb_array"`: Return RGB numpy array

## Action Space

The agent has 7 discrete actions:

| Action | Description |
|--------|-------------|
| 0 | Move Up |
| 1 | Move Down |
| 2 | Move Left |
| 3 | Move Right |
| 4 | Pick up package (if at package location) |
| 5 | Deliver package (if at delivery zone) |
| 6 | No-op (wait) |

## Observation Space

The observation is a normalized float vector containing:
- Robot position (2 values)
- Inventory fill ratio (1 value)
- Package positions (2 × num_packages values)
- Package pickup status (num_packages values)
- Delivery zone positions (2 × num_zones values)
- Obstacle map (grid_height × grid_width values)

## Reward Structure

| Event | Reward |
|-------|--------|
| Each step | -0.1 (encourages efficiency) |
| Pick up package | +10.0 |
| Deliver package | +25.0 |
| Collision (wall/obstacle) | -5.0 |
| Invalid action | -1.0 |

## Configuration

Customize the environment using `WarehouseConfig`:

```python
from openenv_warehouse import WarehouseEnv, WarehouseConfig

config = WarehouseConfig(
    grid_height=15,          # Grid height
    grid_width=15,           # Grid width
    num_packages=5,          # Number of packages
    num_delivery_zones=3,    # Number of delivery zones
    max_steps=300,           # Time limit
    robot_capacity=2,        # Max packages robot can carry
    obstacle_density=0.2,    # Obstacle density (0-0.5)
    seed=42,                 # Random seed
)

env = WarehouseEnv(config)
```

## Visualization

### ASCII Rendering
```
Warehouse - Step: 0/200
Inventory: 0/3
Delivered: 0/3

  0 1 2 3 4 5 6 7 8 9
0 R . . . . . . . . .
1 . . # . . . . . . .
2 . . . . P . . . . .
3 . # . . . . . . . .
4 . . . . . . . . . .
5 . . . . . . . . . .
6 . . . . . . . . . .
7 . . . . . . . . . .
8 . . . . . . . . . D
9 . . . . . . . . . .
```

Legend:
- `R` = Robot
- `P` = Package
- `D` = Delivery zone
- `#` = Obstacle
- `.` = Empty space

### RGB Rendering
The environment can also render as an RGB array for use with visualization libraries or video recording.

## Running Examples

```bash
# Run the basic usage example
python examples/basic_usage.py

# Run tests
python tests/test_environment.py
```

## Project Structure

```
openenv_warehouse/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── spaces.py            # Observation and action spaces
└── environment.py       # Core environment class

examples/
└── basic_usage.py       # Usage examples

tests/
└── test_environment.py  # Unit tests

README.md                # This file
setup.py                 # Package setup (optional)
requirements.txt         # Dependencies
```

## Use Cases

This environment is suitable for:
- **Reinforcement Learning Research**: Test RL algorithms on a practical navigation task
- **AI Agent Training**: Train agents to perform multi-step tasks with sparse rewards
- **Path Planning**: Develop and test path planning algorithms
- **Multi-objective Optimization**: Balance speed, safety, and task completion
- **Educational Purposes**: Teach RL concepts with a tangible, visualizable problem

## Extending the Environment

The environment is designed to be extensible. You can:
- Add new action types (e.g., push obstacles)
- Modify reward structures
- Add multiple robots (multi-agent)
- Change package spawning logic
- Add dynamic obstacles
- Implement curriculum learning with increasing difficulty

## License

MIT License - Feel free to use and modify for your projects.