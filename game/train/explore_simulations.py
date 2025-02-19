import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import math
import random

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers import replay
import matplotlib.pyplot as plt


# Paths
simulations_dir = '../data/'
explored_dir = '../data-explored/'
model_path = Path('../../models/tinyphysics.onnx')

# Predefined torque levels
torques = np.linspace(-2.5, 2.5, 256)

class Controller:
    def __init__(self, epoch=0, internal_pid=None):
        self.epoch = epoch
        self.internal_pid = internal_pid
        self.prev_actions = []
        self.replay_buffer = []

        # Initialize windows for median calculations
        self.steer_window = []
        self.lataccel_window = []

    def store_transition(self, state, action):
        self.replay_buffer.append([state, action, 0.0, 0.0])

    def average(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        v_ego_m_s = max(0, v_ego_m_s)  # Sets negative values to 0
        return math.sqrt(v_ego_m_s) / math.sqrt(max_m_s)

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        global SIM

        # Compute the differences from the current state for each segment
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }

        # Previous steering torque
        previous_action = [0, 0, 0]
        if len(self.prev_actions) >= 3:
            previous_action = self.prev_actions[-3:]

        # Flatten the differences into a single list
        state_input_list = (diff_values['lataccel'] + diff_values['roll'] + diff_values['a_ego'] + diff_values['v_ego'] + previous_action)

        # Prepare the state as input for the model
        state_input = torch.tensor(state_input_list, dtype=torch.float32).unsqueeze(0)

        # Get action from controller
        action = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, steer)

        # Override initial steer commands
        if not math.isnan(steer):
            action = steer

        # Store transition in the replay buffer
        self.store_transition(state_input, action)

        self.prev_actions.append(action)
        return action

def plot_scores(candidate_scores, title="Candidate Torque Scores"):
    """
    Plot candidate torque scores for debugging purposes.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(candidate_scores, marker='o', linestyle='-', color='blue', label='Candidate Torques')

    # Add annotations for each point
    for i, val in enumerate(candidate_scores):
        plt.text(i, val, f"{val:.2f}", fontsize=10, ha='center', va='bottom')

    # Add title, labels, and formatting
    plt.title(title)
    plt.xlabel("Candidate Index")
    plt.ylabel("Score Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)  # Zero reference line
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to simulate and evaluate candidates
def simulate_with_candidates(data_path, torque_data, t, num_candidates=15):
    """Simulate and evaluate candidate torques, returning weighted score differences."""
    # Weights for score deltas
    weights = np.array([0.028, 0.141, 0.831])

    candidate_scores = []
    torque_idx = np.argmin(np.abs(torques - torque_data[t]))  # Find the current torque index
    #print(t, torque_data[t], torque_idx)

    # Dynamically compute the range of torque offsets based on num_candidates
    half_range = (num_candidates - 1) // 2
    candidate_indices = range(max(0, torque_idx - half_range), min(len(torques), torque_idx + half_range + 1))
    candidate_torques = torques[list(candidate_indices)]

    for torque_offset in range(-half_range, half_range + 1):  # Adjust offsets based on num_candidates
        # Modify torque data for the candidate
        modified_torque_data = torque_data.copy()

        for step_offset in range(3):
            torque_idx = np.argmin(np.abs(torques - torque_data[t + step_offset]))
            #print(f"Changed idx {t+step_offset} from {modified_torque_data[t + step_offset]} to {torques[torque_idx + torque_offset]}")
            modified_torque_data[t + step_offset] = torques[torque_idx + torque_offset]

        # Prepare the simulator and controller
        tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
        controller = replay.Controller(torques=modified_torque_data)
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

        step_scores = []
        previous_score = 0
        for step in range(0, t + 4):
            sim.step()
            if step < 84:
                continue

            score = sim.compute_cost()['total_cost']
            step_scores.append(score - previous_score)
            previous_score = score

        # Compute the weighted score using the last 3 steps
        weighted_scores = np.dot(weights, step_scores[-3:])
        candidate_scores.append(weighted_scores)

    #plot_scores(candidate_scores, title="Candidate Torque Scores")
    return candidate_torques, candidate_scores

# Function to explore a single simulation file
def explore_simulation(file_path):
    file_name = os.path.basename(file_path)
    level_num = int(os.path.splitext(file_name)[0])
    data_path = os.path.join('../../data/', f'{level_num:05}.csv')
    explored_file_path = os.path.join(explored_dir, f"{file_name.replace('.npy', '.pth')}")

    # Skip if already explored
    if os.path.exists(explored_file_path):
        print(f"Skipping {file_name}, already explored.")
        return

    # Load torque data
    torque_data = np.load(file_path)
    num_steps = len(torque_data) - 80

    # Prepare the simulator and controller
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
    controller = Controller(internal_pid=replay.Controller(torques=torque_data))
    simulator = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

    training_data = []

    for t in tqdm(range(0, num_steps - 3), desc=f"Exploring {file_name}", disable=False):
        simulator.step()

        if t > 84:
            # Simulate and compute scores for candidates
            candidate_torques, candidate_scores = simulate_with_candidates(data_path, torque_data, t)

            # Store input-output tuple (input_state, candidate_scores)
            candidate_tensor = torch.tensor(candidate_torques, dtype=torch.float32).unsqueeze(0)
            input_state = torch.cat((controller.replay_buffer[-1][0], candidate_tensor), dim=1)
            training_data.append((input_state, candidate_scores))

    # Save explored data using torch
    torch.save(training_data, explored_file_path)

# Main function to explore all simulations
def explore_all_simulations():
    if not os.path.exists(explored_dir):
        os.makedirs(explored_dir)

    files = [os.path.join(simulations_dir, f) for f in os.listdir(simulations_dir) if f.endswith('.npy')]
    shuffled_files = random.sample(files, k=len(files))

    for file_path in shuffled_files:
        explore_simulation(file_path)


if __name__ == "__main__":
    explore_all_simulations()
