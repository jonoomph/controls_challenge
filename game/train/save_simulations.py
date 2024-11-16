import os
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
import matplotlib.pyplot as plt
plt.interactive(False)

import tinyphysics
from controllers import replay, pid, pid_top, pid_future, pid_w_ff

SIM = None
EPOCHS = 1
DATAFILES_START = 0
model_path = Path('../../models/tinyphysics.onnx')


def graph_data(results, rows_per_plot=4):
    """
    Create comparison plots for the top two controllers per file.
    Each row corresponds to one simulation file, with two columns:
    - Left: Lateral acceleration differences
    - Right: Steering outputs
    """
    files = list(results.keys())
    num_files = len(files)
    num_plots = math.ceil(num_files / rows_per_plot)

    for plot_idx in range(num_plots):
        fig, axes = plt.subplots(rows_per_plot, 2, figsize=(12, 3 * rows_per_plot))
        axes = axes.reshape(rows_per_plot, 2)  # Ensure grid shape consistency

        for row_idx, file_idx in enumerate(range(plot_idx * rows_per_plot, min((plot_idx + 1) * rows_per_plot, num_files))):
            file_name = files[file_idx]
            data = results[file_name]

            # Get the top two controllers based on cost
            sorted_scores = sorted(data["scores"].items(), key=lambda x: x[1])
            top_controllers = sorted_scores[:2]

            # Extract controller names and replay buffers for only the top 2 controllers
            controller1_name, controller1_data = top_controllers[0][0], data["buffers"][top_controllers[0][0]]
            controller2_name, controller2_data = top_controllers[1][0], data["buffers"][top_controllers[1][0]]

            # Extract lateral acceleration differences and steering outputs for top 2 controllers
            lataccel_diffs1 = [state[0] for state, _ in controller1_data]
            lataccel_diffs2 = [state[0] for state, _ in controller2_data]
            steer_outputs1 = [action for _, action in controller1_data]
            steer_outputs2 = [action for _, action in controller2_data]

            # Left column: Plot lateral acceleration differences for top 2 controllers
            ax = axes[row_idx, 0]
            ax.plot(lataccel_diffs1, label=f"{controller1_name}")
            ax.plot(lataccel_diffs2, label=f"{controller2_name}")
            ax.set_title(f"File: {file_name} - Lat Accel Diff")
            ax.set_xlabel("Time")
            ax.set_ylabel("Lateral Accel Diff")
            ax.legend()

            # Right column: Plot steering outputs for top 2 controllers
            ax = axes[row_idx, 1]
            ax.plot(steer_outputs1, label=f"{controller1_name}")
            ax.plot(steer_outputs2, label=f"{controller2_name}")
            ax.set_title(f"File: {file_name} - Steering Outputs")
            ax.set_xlabel("Time")
            ax.set_ylabel("Steering")
            ax.legend()

        #plt.tight_layout()
        plt.show()



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
        self.replay_buffer.append((state, action))

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
                          4959, 4743, 4744, 3166, 2732, 3712, 1722, 4620, 3163, 2531, 3917, 3132, 1193, 264, 2105, 4803, 526, 2684, 1805, 2108 ]:
        file_list.append(f"{missing_level:05}.npy")

    results = defaultdict(dict)
    win_counts = defaultdict(int)

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
                sim.rollout()
                cost = sim.compute_cost()
                scores[controller_name] = cost["total_cost"]
                replay_buffers[controller_name] = sim.controller.replay_buffer

            # Determine the best controller
            best_controller_name = min(scores, key=scores.get)
            best_score = scores[best_controller_name]
            best_model_data = replay_buffers[best_controller_name]

            # Save the best controller's data
            save_path = f'simulations-combined/{level_num:05d}-{best_controller_name}.pth'
            torch.save(best_model_data, save_path)

            # Also save PID-REPLAY (if close to best score)
            # replay_controller_name = "PID-REPLAY"
            # if best_controller_name != replay_controller_name and abs(scores[best_controller_name] - scores[replay_controller_name]) < 10.0:
            #     save_path = f'simulations-combined/{level_num:05d}-{replay_controller_name}.pth'
            #     torch.save(replay_buffers[replay_controller_name], save_path)
            #     win_counts[replay_controller_name] += 1

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

    # Plot the data
    #graph_data(results)


if __name__ == "__main__":
    start_training()
