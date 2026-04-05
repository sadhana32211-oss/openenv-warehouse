#!/usr/bin/env python3
"""
Hugging Face Space app for OpenEnv Warehouse environment.
Interactive demo of warehouse robot navigation for AI agents.
"""

import gradio as gr
import numpy as np
from openenv_warehouse import WarehouseEnv


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

if __name__ == "__main__":
    demo.launch()