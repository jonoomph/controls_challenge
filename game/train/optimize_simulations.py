import os
import numpy as np
import copy
from controllers import replay
import matplotlib.pyplot as plt
import tinyphysics
from pathlib import Path
model_path = Path('../../models/tinyphysics.onnx')
tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

# Define the path to the simulations directory
from tqdm import tqdm

simulations_dir = '../data/'
solved_dir = '../data-optimized/'


class Controller:
    def __init__(self, replay_data=None, largest_index=0, threshold=.1):
        self.prev_state = None
        self.prev_action = 0
        self.internal_pid = replay.Controller(torques=replay_data)
        self.replay_buffer = []
        self.stopped = False
        self.diff = 0
        self.largest_index = largest_index
        self.threshold = threshold

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=None):
        action = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, steer)

        step = self.internal_pid.step_idx - 1
        #index = step - 20
        # if self.prev_action:
        #     self.diff = target_lataccel - current_lataccel
        #     if abs(self.diff) > self.threshold and step >= 110 and index >= self.largest_index:
        #         self.largest_index = index
        #         self.stopped = True
        self.prev_action = action
        return action

def run_simulation(data_path, tensor_data, largest_index, threshold):
    # Create simulator
    SIM = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=Controller(replay_data=tensor_data, largest_index=largest_index, threshold=threshold), debug=False)

    # Iterate through data rows (where steering was active)
    for _ in range(20, len(SIM.data)):
        SIM.step()
        if SIM.controller.stopped:
            return SIM.controller.largest_index, SIM.controller.diff, SIM.controller.prev_action, SIM.compute_cost()['total_cost']
    return 579, 0.0, 0.0, SIM.compute_cost()['total_cost']

def plot_adjusted_vs_original_torque(original_torque, adjusted_torque):
    """
    Plots the original torque, adjusted torque, and the fitted spline, along with the accel differences.

    :param original_torque: The original torque values.
    :param adjusted_torque: The adjusted torque values before spline fitting.
    """
    steps = np.arange(len(original_torque))

    plt.figure(figsize=(12, 8))

    # Plot original torque
    plt.plot(steps, original_torque, label='Original Torque', color='blue', linewidth=1.5)

    # Plot final smoothed torque
    plt.plot(steps, adjusted_torque, label='Adjusted Torque (Solved)', color='red', linewidth=1.5)

    plt.xlabel('Step')
    plt.ylabel('Value / %')
    plt.title('Torque Comparison and Lateral Acceleration Difference (as %)')
    plt.legend()
    plt.grid(True)
    plt.show()

def optimize_inside_sim(file_name, mode, tensor_data):
    data_file = os.path.splitext(file_name)[0]
    data_path = os.path.join("../../data", f"{data_file}.csv")
    original_data = copy.deepcopy(tensor_data)

    # Get initial total cost
    _, _, _, lowest_cost = run_simulation(data_path, tensor_data, largest_index=0, threshold=0)
    print(f"Optimize {data_file} (starting cost: {lowest_cost})")

    num_iterations = 2
    index_range = 4

    # Define the discrete torque levels
    torques = np.linspace(-2.5, 2.5, 256)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        pbar = tqdm(range(80, len(tensor_data)), dynamic_ncols=True)
        for idx in pbar:
            current_torque = tensor_data[idx]

            # Find the index in torques closest to current_torque
            torque_idx = np.argmin(np.abs(torques - current_torque))

            # Define the range of torque indices to try
            candidate_indices = np.arange(torque_idx - index_range, torque_idx + index_range + 1)

            # Ensure indices are within bounds
            candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < len(torques))]

            # Run simulation
            _, _, _, cost = run_simulation(data_path, tensor_data, largest_index=0, threshold=0)
            pbar.set_postfix(cost=cost)
            best_torque = current_torque
            best_cost = cost

            for candidate_idx in candidate_indices:
                adjusted_torque = torques[candidate_idx]

                # Create a copy of tensor_data with adjusted torque at idx
                tensor_data_adjusted = tensor_data.copy()
                tensor_data_adjusted[idx] = adjusted_torque

                # Run simulation
                _, _, _, cost = run_simulation(data_path, tensor_data_adjusted, largest_index=0, threshold=0)

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_torque = adjusted_torque
                    pbar.set_postfix(cost=cost)

            # Update tensor_data with best_torque
            tensor_data[idx] = best_torque

        # After iterating over dataset once, get total cost
        _, _, _, total_cost = run_simulation(data_path, tensor_data, largest_index=0, threshold=0)
        print(f"Total cost after iteration {iteration+1}: {total_cost}")
        if total_cost < lowest_cost:
            print(f"Improved total cost from {lowest_cost} to {total_cost}")
            lowest_cost = total_cost
            # Save optimized data
            if mode == 'optimize':
                solved_file_name = f"{data_file}.npy"  # Save as .npy since tensor_data is a numpy array
                output_path = os.path.join(solved_dir, solved_file_name)
                np.save(output_path, tensor_data)
        else:
            print(f"No improvement in total cost. Stopping optimization.")
            # Optionally, break out of the loop if no improvement
            break

    # Optionally, plot the torque adjustments
    if mode == 'optimize':
        # Plot the original torque data with the adjusted torque
        original_torque = np.array([row[1] for row in original_data])
        adjusted_torque = np.array([row[1] for row in tensor_data])
        plot_adjusted_vs_original_torque(original_torque, adjusted_torque)

def optimize_simulations(mode='print'):
    # Ensure the solved directory exists
    if not os.path.exists(solved_dir):
        os.makedirs(solved_dir)

    # Loop through each file in the simulations directory
    for filename in sorted(os.listdir(simulations_dir)):
        if filename.endswith('.npy'):  # Assuming tensors are saved with a .pt extension
            file_path = os.path.join(simulations_dir, filename)
            solved_path = os.path.join(solved_dir, filename)

            # Load the tensor from the file
            if not os.path.exists(solved_path):
                tensor_data = np.load(file_path)

                # Optimize or print the actions in the tensor_data
                optimize_inside_sim(filename, mode, tensor_data)


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optimize or print simulation data.")
    parser.add_argument('--mode', type=str, default='print', choices=['print', 'optimize'],
                        help="Mode: 'print' to display ratios and predictions, 'optimize' to save optimized tensors.")
    args = parser.parse_args()

    # Run the optimization script
    optimize_simulations(args.mode)
