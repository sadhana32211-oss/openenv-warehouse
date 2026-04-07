#!/usr/bin/env python3
"""
Hugging Face Space app for OpenEnv Warehouse environment.
Interactive demo of warehouse robot navigation for AI agents.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path to find openenv_warehouse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from openenv_warehouse import WarehouseEnv, WarehouseConfig
    HAS_ENVIRONMENT = True
except ImportError:
    HAS_ENVIRONMENT = False
    print("Warning: openenv_warehouse module not found, using mock data")

# Simple text-based interface since Gradio has issues
def run_simulation(num_steps=50, grid_size=10, num_packages=3):
    """Run warehouse robot simulation."""
    config = WarehouseConfig(
        grid_height=grid_size, grid_width=grid_size, 
        num_packages=num_packages, max_steps=num_steps, seed=42
    )
    env = WarehouseEnv(config)
    env.reset()
    
    frames = ["=== OpenEnv Warehouse Simulation ==="]
    total_reward = 0
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            frame = env.render(mode="ansi")
            frames.append(f"Step {step + 1}: Action {action} | Reward: {reward:+.1f}")
            frames.append(frame)
            frames.append(f"Delivered: {info['packages_delivered']}/{info['total_packages']}")
            frames.append("-" * 50)
        
        if terminated or truncated:
            break
    
    return "\n".join(frames)

# Create simple HTML interface
def create_html_interface():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenEnv Warehouse - Robot Navigation Demo</title>
        <style>
            body { font-family: monospace; background: #f0f0f0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #333; }
            .controls { margin: 20px 0; }
            label { display: block; margin: 10px 0 5px; }
            input[type=range] { width: 100%; }
            button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #output { background: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; min-height: 300px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 OpenEnv Warehouse Environment</h1>
            <p>Interactive demo of warehouse robot navigation for AI agents.</p>
            
            <div class="controls">
                <label>Steps: <span id="stepsVal">50</span></label>
                <input type="range" id="steps" min="10" max="200" value="50">
                
                <label>Grid Size: <span id="gridVal">10</span></label>
                <input type="range" id="grid" min="5" max="20" value="10">
                
                <label>Packages: <span id="pkgVal">3</span></label>
                <input type="range" id="packages" min="1" max="10" value="3">
                
                <br><br>
                <button onclick="runSimulation()">🚀 Run Simulation</button>
            </div>
            
            <div id="output">Click "Run Simulation" to start...</div>
        </div>
        
        <script>
            document.getElementById('steps').oninput = function() {
                document.getElementById('stepsVal').textContent = this.value;
            };
            document.getElementById('grid').oninput = function() {
                document.getElementById('gridVal').textContent = this.value;
            };
            document.getElementById('packages').oninput = function() {
                document.getElementById('pkgVal').textContent = this.value;
            };
            
            async function runSimulation() {
                const steps = document.getElementById('steps').value;
                const grid = document.getElementById('grid').value;
                const packages = document.getElementById('packages').value;
                
                document.getElementById('output').textContent = 'Running simulation...';
                
                // Since we can't run the actual Python code in browser,
                // we'll show a mock simulation
                const mockOutput = `=== OpenEnv Warehouse Simulation ===
Grid Size: ${grid}x${grid}
Packages: ${packages}
Max Steps: ${steps}

Initial State:
Warehouse - Step: 0/${steps}
Inventory: 0/3
Delivered: 0/${packages}

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

Step 5: Action 3 | Reward: +0.0
Warehouse - Step: 5/${steps}
Inventory: 0/3
Delivered: 0/${packages}

  0 1 2 3 4 5 6 7 8 9
0 . R . . . . . . . .
1 . . # . . . . . . .
2 . . . . P . . . . .
3 . # . . . . . . . .
4 . . . . . . . . . .
5 . . . . . . . . . .
6 . . . . . . . . . .
7 . . . . . . . . . .
8 . . . . . . . . . D
9 . . . . . . . . . .

Delivered: 0/${packages}
--------------------------------------------------
Step 10: Action 4 | Reward: -1.1
Warehouse - Step: 10/${steps}
Inventory: 0/3
Delivered: 0/${packages}

  0 1 2 3 4 5 6 7 8 9
0 . R . . . . . . . .
1 . . # . . . . . . .
2 . . . . P . . . . .
3 . # . . . . . . . .
4 . . . . . . . . . .
5 . . . . . . . . . .
6 . . . . . . . . . .
7 . . . . . . . . . .
8 . . . . . . . . . D
9 . . . . . . . . . .

Delivered: 0/${packages}
--------------------------------------------------

=== Simulation Summary ===
Steps Taken: 50
Total Reward: -5.0
Packages Delivered: 0/${packages}
Collisions: 2`;
                
                document.getElementById('output').textContent = mockOutput;
            }
        </script>
    </body>
    </html>
    """
    return html

# For Hugging Face Space, we'll use a simple approach
if __name__ == "__main__":
    # Try to import gradio, but fall back to simple HTML if it fails
    try:
        import gradio as gr
        
        # Create Gradio interface
        with gr.Blocks(title="OpenEnv Warehouse") as demo:
            gr.Markdown("# 🤖 OpenEnv Warehouse Environment")
            gr.Markdown("Interactive demo of warehouse robot navigation for AI agents.")
            
            with gr.Row():
                num_steps = gr.Slider(10, 200, value=50, label="Steps")
                grid_size = gr.Slider(5, 20, value=10, label="Grid Size")
                num_packages = gr.Slider(1, 10, value=3, label="Packages")
            
            run_button = gr.Button("🚀 Run Simulation", variant="primary")
            output = gr.Textbox(label="Output", lines=20)
            
            run_button.click(run_simulation, [num_steps, grid_size, num_packages], output)
        
        demo.launch()
    except ImportError:
        # Fallback to simple HTML
        print("Content-Type: text/html\n")
        print(create_html_interface())
