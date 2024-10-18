import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import tinyphysics
from controllers.pid_model import Controller

model_path = Path('../../models/tinyphysics.onnx')
DATAFILES_START = 0
DATAFILES_LENGTH = 20
MAX_TRAINING_ROWS = 600


def start_testing(filter=None):
    # Setup SummaryWriter for TensorBoard logging
    global MAX_TRAINING_ROWS, model_path
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    model_costs = defaultdict(lambda: {"total_cost": 0, "file_count": 0})

    for model_name in tqdm(sorted(os.listdir("onnx"))):
        model_path = os.path.abspath(os.path.join("onnx", model_name))
        if os.path.isdir(model_path) or (filter and filter not in model_name):
            continue

        for file_index in range(DATAFILES_START, DATAFILES_START + DATAFILES_LENGTH):
            data_path = f'../../data/{file_index:05d}.csv'

            # Create simulator
            controller = Controller(model_path=model_path)
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
    print(f"Top 5 Best Performing Models (Average Cost x {DATAFILES_LENGTH}):")
    for i in range(len(average_costs)):
        model_name, avg_cost = average_costs[i]
        print(f"#{i+1} Model: {model_name}, Average Cost: {avg_cost:.4f}")


if __name__ == "__main__":
    start_testing()
