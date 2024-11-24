import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import tinyphysics
from controllers.pid_model import Controller


DATAFILES_START = 0
MAX_TRAINING_ROWS = 600
TRAINING_FILES = [3585, 4120, 105, 4656, 765, 4135, 1556, 4580, 2818, 4149, 3923, 1658, 3094, 769, 1415, 2672, 4002,
                  4169, 3257, 3755, 2688, 4730, 4162, 1673, 3012, 3775]

def start_testing(filter=None, logging=True, window_size=22, training_files=26):
    # Setup SummaryWriter for TensorBoard logging
    global MAX_TRAINING_ROWS
    tiny_model_path = Path('../../models/tinyphysics.onnx')
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(tiny_model_path, debug=False)

    model_costs = defaultdict(lambda: {"total_cost": 0, "file_count": 0})

    for model_name in tqdm(sorted(os.listdir("onnx")), disable=not logging):
        model_path = os.path.abspath(os.path.join("onnx", model_name))
        if os.path.isdir(model_path) or (filter and filter not in model_name) or not model_path.endswith(".onnx"):
            continue

        for file_index in range(DATAFILES_START, DATAFILES_START + training_files):
            TRAINING_FILE = TRAINING_FILES[file_index]
            data_path = f'../../data/{TRAINING_FILE:05d}.csv'

            # Create simulator
            controller = Controller(model_path=model_path, window_size=window_size)
            sim = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)
            sim.rollout()
            cost = sim.compute_cost()

            # Accumulate cost and file count for the current model
            model_costs[model_name]["total_cost"] += cost["total_cost"]
            model_costs[model_name]["file_count"] += 1

    # Calculate average cost per model
    average_costs = [(model_name, values["total_cost"] / values["file_count"])
                     for model_name, values in model_costs.items()]

    # Sort models by average cost (ascending order)
    average_costs.sort(key=lambda x: x[1])

    # Print top 5 best-performing models
    if logging:
        for i in range(len(average_costs)):
            model_name, avg_cost = average_costs[i]
            print(f"#{i+1} Model: {model_name}, Average Cost: {avg_cost:.4f}")

    # Return top cost
    if average_costs:
        return average_costs[0][1]
    else:
        return 0.0

if __name__ == "__main__":
    start_testing()
