# train.py
import string

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from pathlib import Path

from tqdm import tqdm

from qmodel import QSteeringNet
import simulations
import config

# Set device and seeds.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Global replay buffer (holds transitions from simulations)
replay_buffer = deque(maxlen=config.replay_buffer_size)


def sample_batch(buffer, batch_size):
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    # For next_states, if a transition is missing next_state, use a zero tensor.
    next_states_tensor = []
    for ns in next_states:
        if ns is None:
            next_states_tensor.append(torch.zeros(1, config.window_size, config.input_size))
        else:
            # Assume ns already has shape (1, input_size); expand to (1, window_size, input_size) if needed.
            next_states_tensor.append(ns)
    next_states = torch.cat(next_states_tensor).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    return states, actions, rewards, next_states, dones


def export_model(epoch=None, prefix="model", window_size=7, model=None, logging=True):
    # Save the trained model to an ONNX file
    model_name = f"onnx/model-{prefix}-{epoch}.onnx"
    if logging:
        print(f"Exporting model: {model_name}")

    # Adjust the dummy input size according to the model's expected input size
    dummy_input = torch.randn(1, window_size, model.input_size, requires_grad=True).to(device)

    try:
        # Export the model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            model_name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'window_size'}, 'output': {0: 'batch_size'}}
        )
    except Exception as e:
        print(f"Error during ONNX export for epoch {epoch}: {e}")
        raise

def main():
    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))

    # Create Q-network.
    q_network = QSteeringNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        window_size=config.window_size,
        action_space=config.action_space
    ).to(device)
    target_network = QSteeringNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        window_size=config.window_size,
        action_space=config.action_space
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}")
        # Run simulations to collect transitions.
        sim_results = simulations.get_simulations(q_network, num_sims=config.num_sims, config=config, current_epoch=epoch)
        for sim_data in tqdm(sim_results.values()):
            replay_buffer.clear()
            for transition in sim_data["buffer"][80:-80]:
                replay_buffer.append(transition)

            # Training: multiple updates per epoch
            epoch_loss = 0
            for mini_batch in range(config.num_mini_batches):
                # Training: only if enough samples are available.
                if len(replay_buffer) >= config.batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_batch(replay_buffer, config.batch_size)

                    # Compute current Q-values for the chosen actions.
                    current_q = q_network(batch_states).gather(1, batch_actions)
                    with torch.no_grad():
                        next_q = target_network(batch_next_states)
                        max_next_q, _ = torch.max(next_q, dim=1, keepdim=True)

                        rescaled_rewards = (torch.clamp(batch_rewards, -5.0, 5.0) / 5.0)
                        target_q = rescaled_rewards + (config.gamma * max_next_q * (1 - batch_dones))

                    loss = loss_fn(current_q, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 0.5)
                    optimizer.step()
                    epoch_loss += loss.item()

        target_network.load_state_dict(q_network.state_dict())
        print(f"Training Loss - {epoch}: {epoch_loss / config.num_mini_batches}")

        # Export the model at intervals.
        if epoch % config.export_interval == 0:
            export_model(epoch, prefix, config.window_size, q_network, optimizer)


if __name__ == "__main__":
    main()
