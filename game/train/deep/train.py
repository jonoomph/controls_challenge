# train.py
import string

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
q_values_history = []
sample_state = torch.randn(1, config.window_size, config.input_size).to(device)



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

    # Save torch model
    torch.save(model.state_dict(), model_name.replace(".onnx", ".pth"))
    return

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

def plot_data(epoch, replay_buffer):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract rewards from replay buffer
    rewards = [transition[2] for transition in replay_buffer if transition[2] is not None]

    # Basic stats
    print("Transformed Reward stats:",
          "min =", min(rewards),
          "max =", max(rewards),
          "mean =", np.mean(rewards))

    # Plot histogram of transformed rewards
    plt.figure(figsize=(8, 6))
    plt.hist(rewards, bins=20, edgecolor='black')
    plt.xlabel("Rewards")
    plt.ylabel("Frequency")
    plt.title(f"Reward Distribution at Epoch {epoch}")
    plt.show()


def plot_q_values(q_network, sample_state, epoch, q_values_history):
    """
    Plot Q-values for all 257 actions across training epochs.

    Args:
        q_network: Trained Q-network
        sample_state: Fixed sample input state (for consistent comparison)
        epoch: Current epoch number
        q_values_history: List to store Q-values across epochs
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get Q-values for all actions using current network
    with torch.no_grad():
        q_values = q_network(sample_state).cpu().numpy().flatten()

    # Store in history (transposed for plotting)
    q_values_history.append(q_values)

    # Convert to numpy array
    q_array = np.array(q_values_history)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot heatmap
    plt.imshow(q_array.T,  # Transpose to get actions on y-axis
               aspect='auto',
               cmap='viridis',
               origin='lower')

    # Formatting
    plt.title(f"Q-value Distribution Across Actions (Epoch {epoch})")
    plt.xlabel("Training Epoch")
    plt.ylabel("Action Index")
    plt.colorbar(label="Q-value")

    # Action axis formatting
    plt.yticks(np.arange(0, 257, 50), labels=np.arange(0, 257, 50))

    # Epoch axis formatting
    if epoch > 10:
        plt.xticks(np.arange(0, epoch, max(1, epoch // 10)))

    plt.tight_layout()
    plt.show()

def train(replay_buffer, q_network, target_network, loss_fn, optimizer, validate=False):
    # Training: multiple updates per epoch
    total_loss =  0.0
    for mini_batch in range(config.num_mini_batches):
        # Training: only if enough samples are available.
        if len(replay_buffer) >= config.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_batch(replay_buffer, config.batch_size)

            # Compute current Q-values for the chosen actions.
            current_q = q_network(batch_states).gather(1, batch_actions)
            with torch.no_grad():
                next_q = target_network(batch_next_states)
                max_next_q, _ = torch.max(next_q, dim=1, keepdim=True)
                target_q = batch_rewards + (config.gamma * max_next_q)

            # Loss and backprop
            loss = loss_fn(current_q, target_q)
            total_loss += loss.item()

            if not validate:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 0.5)
                optimizer.step()

            # After computing current_q and next_q:
            # with torch.no_grad():
            #     current_q_values = q_network(batch_states)  # Q-values for all actions
            #     max_current_q = torch.max(current_q_values).item()  # Max Q-value for current states
            #     max_next_q = torch.max(next_q).item()  # Max Q-value for next states (from target network)
            #     # print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Max Current Q: {max_current_q:.2f}, Max Next Q: {max_next_q:.2f}")

    # Soft target network update
    tau = 0.05  # Mixing factor: 5% new weights, 95% old target weights
    with torch.no_grad():
        for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * q_param.data)

    return total_loss / config.num_mini_batches

def main():
    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))

    # Create Q-network.
    q_network = QSteeringNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        window_size=config.window_size
    ).to(device)
    target_network = QSteeringNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        window_size=config.window_size
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss()

    # Get Validation data
    print("Get Validation Data")
    validate_buffer = []
    validate_results = simulations.get_simulations(q_network, num_sims=config.num_sims, config=config, current_epoch=-1, max_noise=0)
    for val_data in tqdm(validate_results.values()):
        replay_buffer.clear()
        for transition in val_data["buffer"][80:-80]:
            validate_buffer.append(transition)

    #plot_data(-1, validate_buffer)

    max_noise = config.max_noise
    for epoch in range(config.epochs):
        print(f"Epoch {epoch}")
        # Run simulations to collect transitions.
        sim_results = simulations.get_simulations(q_network, num_sims=config.num_sims, config=config, current_epoch=epoch, max_noise=max_noise)
        epoch_loss = 0
        validate_loss = 0

        for sim_data in tqdm(sim_results.values()):
            replay_buffer.clear()
            for transition in sim_data["buffer"][80:-80]:
                replay_buffer.append(transition)

            # Train on replay buffer
            epoch_loss += train(replay_buffer, q_network, target_network, loss_fn, optimizer)

            # Validate
            validate_loss += train(validate_buffer, q_network, target_network, loss_fn, optimizer, validate=True)
        print(f"Training Loss : {epoch_loss / config.num_mini_batches}, Validation Loss : {validate_loss / config.num_mini_batches}")

        # adjust noise
        if max_noise > 0.0:
            max_noise -= 0.25
        else:
            # reset noise
            max_noise = config.max_noise

        # Export the model at intervals.
        if epoch % config.export_interval == 0:
            export_model(epoch, prefix, config.window_size, q_network, optimizer)
            #plot_q_values(q_network, sample_state, epoch, q_values_history)


if __name__ == "__main__":
    main()
