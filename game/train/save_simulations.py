import os
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.interactive(False)

import tinyphysics
from controllers import replay, pid, pid_top, pid_future, pid_w_ff

SIM = None
EPOCHS = 1
DATAFILES_START = 0
model_path = Path('../../models/tinyphysics.onnx')


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
        self.replay_buffer.append([state, action, 0.0])

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
        future_segments = [(0, 1), (1, 3), (2, 5)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [state.a_ego - self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
            'lataccel_roll': [current_lataccel - (self.average(future_plan.lataccel[start:end]) -
                                                  self.average(future_plan.roll_lataccel[start:end])) for start, end in future_segments],
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
        else:
            # Store transition in the replay buffer
            self.store_transition(state_input, action)

        self.prev_actions.append(action)
        return action

def get_random_files(folder_path, num_files=1, seed=1):
    # Set the seed for repeatability
    random.seed(seed)

    # Get a list of all files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".npy")]

    # If there are fewer files than the requested number, return all files
    if len(all_files) <= num_files:
        return all_files

    # Randomly sample the files
    sampled_files = random.sample(all_files, num_files)

    return sampled_files

def start_training(num_files=99, threshold=1.0):
    global SIM
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    # Get random files
    file_list = get_random_files('../data-optimized/', num_files=num_files, seed=1979)

    # Append missing levels
    for missing_level in [2589, 3001, 2580, 432, 4772, 1954, 1550, 2618, 739, 1483, 3385, 660, 3079, 367, 2898, 4686, 4877, 2879, 1202, 699,
                          4743, 2732, 3712, 4620, 3163, 3917, 3132, 264, 2105, 526, 2684, 1805, 2108]:
        file_list.append(f"{missing_level:05}.npy")

    results = defaultdict(dict)
    win_counts = defaultdict(int)
    all_steer_costs = []

    for epoch in range(EPOCHS):
        for file_name in file_list:
            level_num = int(os.path.splitext(file_name)[0])
            data_path = os.path.join('../../data/', f'{level_num:05}.csv')

            # Dictionary to store scores and replay buffers
            scores = {}
            replay_buffers = {}

            # Run simulations with each controller and store scores
            for controller_name, internal_controller in [
                ("PID-REPLAY", replay.Controller(level_num=level_num)),
                ("PID-TOP", pid_top.Controller()),
                ("PID-FF", pid_w_ff.Controller()),
                ("PID-FUTURE", pid_future.Controller()),
                ("PID", pid.Controller())
            ]:
                # Create simulator with the specific controller
                sim = tinyphysics.TinyPhysicsSimulator(
                    tinyphysicsmodel, str(data_path),
                    controller=Controller(epoch, internal_controller),
                    debug=False
                )

                # Run the simulation and calculate the cost
                previous_cost = 0.0
                for _ in range(20, len(sim.data)):
                    sim.step()

                    if _ >= 101:
                        total_cost = sim.compute_cost().get('total_cost')
                        if not math.isnan(total_cost):
                            sim.controller.replay_buffer[-1][2] = total_cost - previous_cost
                            all_steer_costs.append(total_cost - previous_cost)
                            previous_cost = total_cost

                cost = sim.compute_cost()
                scores[controller_name] = cost["total_cost"]
                replay_buffers[controller_name] = sim.controller.replay_buffer

            # Determine the best controller
            best_controller_name = min(scores, key=scores.get)
            best_score = scores[best_controller_name]
            best_model_data = replay_buffers[best_controller_name]

            # Save the best controller's data
            save_path = f'simulations/{level_num:05d}-{best_controller_name}.pth'
            torch.save(best_model_data, save_path)

            # Update win counts
            win_counts[best_controller_name] += 1

            # Save data for plotting
            results[file_name] = {"scores": scores, "buffers": replay_buffers}

            # Output the score comparison
            score_diffs = [f"{name}: {scores[name] - best_score:+.1f}" for name in scores if name != best_controller_name]
            score_output = f"{level_num:05d}: {best_controller_name} : {best_score:.1f} cost ({', '.join(score_diffs)})"
            print(score_output)

    # Final breakdown of wins per controller
    print("\nFinal Summary: Number of Wins per Controller")
    for controller_name, count in win_counts.items():
        print(f"{controller_name}: {count} wins")

    # Compute percentiles to identify outliers
    lower_percentile = np.percentile(all_steer_costs, 5)  # 5th percentile
    upper_percentile = np.percentile(all_steer_costs, 95)  # 95th percentile

    # Filter steer costs within the 5th and 95th percentile
    trimmed_steer_costs = [x for x in all_steer_costs if lower_percentile <= x <= upper_percentile]

    # Plot the trimmed distribution for better focus
    plt.figure(figsize=(12, 6))
    plt.hist(trimmed_steer_costs, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=np.mean(trimmed_steer_costs), color='red', linestyle='--',
                label=f'Mean (Trimmed): {np.mean(trimmed_steer_costs):.2f}')
    plt.axvline(x=np.median(trimmed_steer_costs), color='green', linestyle='--',
                label=f'Median (Trimmed): {np.median(trimmed_steer_costs):.2f}')
    plt.title('Trimmed Distribution of Steer Costs (5th to 95th Percentile)')
    plt.xlabel('Steer Cost')
    plt.ylabel('Frequency')
    plt.legend()


    plt.show()
if __name__ == "__main__":
    start_training()
