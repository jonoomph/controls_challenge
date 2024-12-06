from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from pathlib import Path
import tinyphysics
from controllers import replay


# Define paths
simulations_dir = '../data/'
solved_dir = '../data-optimized/'
model_path = Path('../../models/tinyphysics.onnx')
torques = np.linspace(-2.5, 2.5, 256)


# Controller class definition
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
        self.prev_action = action
        return action

# Function to run a single simulation
def run_simulation(data_path, tensor_data, largest_index, threshold, model_path):
    # Create thread-local TinyPhysicsModel and Controller instance
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)
    controller = Controller(replay_data=tensor_data, largest_index=largest_index, threshold=threshold)

    SIM = tinyphysics.TinyPhysicsSimulator(
        tinyphysicsmodel,
        str(data_path),
        controller=controller,  # Pass a unique Controller instance
        debug=False
    )

    for _ in range(20, len(SIM.data)):
        SIM.step()
        if SIM.controller.stopped:
            return SIM.controller.largest_index, SIM.controller.diff, SIM.controller.prev_action, SIM.compute_cost()['total_cost']
    return 579, 0.0, 0.0, SIM.compute_cost()['total_cost']

# Function to optimize torque adjustments
def optimize_inside_sim(file_path, mode, progress_position):
    file_name = os.path.basename(file_path)
    data_file = os.path.splitext(file_name)[0]
    data_path = os.path.join("../../data", f"{data_file}.csv")
    tensor_data = np.load(file_path)

    _, _, _, lowest_cost = run_simulation(data_path, tensor_data.copy(), largest_index=0, threshold=0, model_path=model_path)

    num_iterations = 1
    index_range = 2

    for iteration in range(num_iterations):
        with tqdm(range(80, len(tensor_data) - 80), desc=f"Optimizing {file_name}", position=1, leave=False, disable=False) as pbar:
            for idx in pbar:
                current_torque = tensor_data[idx]
                torque_idx = np.argmin(np.abs(torques - current_torque))
                candidate_indices = np.arange(torque_idx - index_range, torque_idx + index_range + 1)
                candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < len(torques))]

                # Remove the current torque index from candidate indices
                candidate_indices = candidate_indices[candidate_indices != torque_idx]

                # Run simulation for the current state to get the baseline cost
                _, _, _, cost = run_simulation(data_path, tensor_data.copy(), largest_index=0, threshold=0, model_path=model_path)
                best_torque = current_torque
                best_cost = cost

                for candidate_idx in candidate_indices:
                    adjusted_torque = torques[candidate_idx]
                    tensor_data_adjusted = tensor_data.copy()
                    tensor_data_adjusted[idx] = adjusted_torque

                    _, _, _, cost = run_simulation(data_path, tensor_data_adjusted, largest_index=0, threshold=0, model_path=model_path)

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_torque = adjusted_torque

                tensor_data[idx] = best_torque
                pbar.set_postfix(orig_cost=lowest_cost, new_cost=best_cost, diff=lowest_cost - best_cost, data=data_path)
                #print(f"{data_file}: best_cost: {best_cost}, lowest_cost: {lowest_cost}")

        _, _, _, total_cost = run_simulation(data_path, tensor_data, largest_index=0, threshold=0, model_path=model_path)
        print(f"Total cost after iteration {iteration+1}: {total_cost}")
        if total_cost < lowest_cost:
            lowest_cost = total_cost
            if mode == 'optimize':
                solved_file_name = f"{data_file}.npy"
                output_path = os.path.join(solved_dir, solved_file_name)
                np.save(output_path, tensor_data)
        else:
            break

# Main optimization function with threading
def optimize_simulations(mode='optimize', file_offset=0):
    if not os.path.exists(solved_dir):
        os.makedirs(solved_dir)

    files = sorted([f for f in os.listdir(simulations_dir) if f.endswith('.npy')])[file_offset:]
    solved_files = sorted([f for f in os.listdir(solved_dir) if f.endswith('.npy')])[file_offset:]
    file_paths = [os.path.join(simulations_dir, f) for f in files if f not in solved_files]

    with ThreadPoolExecutor(max_workers=7) as executor:
        with tqdm(total=len(file_paths), desc="Optimizing simulations", position=0, leave=False, disable=False) as outer_pbar:
            futures = [
                executor.submit(optimize_inside_sim, file_path, mode, i + 1)
                for i, file_path in enumerate(file_paths)
            ]
            for future in futures:
                future.result()
                outer_pbar.update(1)

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optimize or print simulation data.")
    parser.add_argument('--mode', type=str, default='optimize', choices=['optimize', 'print'],
                        help="Mode: 'print' to display ratios and predictions, 'optimize' to save optimized tensors.")
    parser.add_argument('--file_offset', type=int, default=0,
                        help="File Offset: offset the starting file index.")
    args = parser.parse_args()

    # Run the optimization script
    optimize_simulations(args.mode, args.file_offset)
