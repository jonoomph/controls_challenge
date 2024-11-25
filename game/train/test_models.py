import os
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import tinyphysics
from controllers.pid_model import Controller

DATAFILES_START = 0
MAX_TRAINING_ROWS = 600
TRAINING_FILES = [3373, 3727, 2456, 955, 2605, 4200, 2134, 2934, 4505, 584, 3656, 3804, 3774, 2157, 2281, 2116, 3545, 2141, 4859, 2903, 2197, 4871, 3564, 3576, 2224, 3281, 2326, 3481, 3519, 4085, 2399, 4872, 2430, 4050, 4067, 4251, 1226, 3217, 1784, 2920, 3361, 2757, 1048, 1781, 1046, 4347, 3091, 4574, 3886, 2506, 4045, 805, 2222, 3956, 4588, 3149, 3580, 1180, 4732, 3003]

def process_file(model_name, model_path, file_index, window_size, tinyphysicsmodel):
    try:
        TRAINING_FILE = TRAINING_FILES[file_index]
        data_path = f'../../data/{TRAINING_FILE:05d}.csv'

        # Create simulator
        controller = Controller(model_path=model_path, window_size=window_size)
        sim = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()

        return model_name, cost["total_cost"]
    except Exception as e:
        print(f"Error processing file {TRAINING_FILE}: {e}")
        return model_name, float("inf")

def start_testing(filter=None, logging=True, window_size=22, training_files=60):
    global MAX_TRAINING_ROWS
    tiny_model_path = Path('../../models/tinyphysics.onnx')
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(tiny_model_path, debug=False)

    model_costs = defaultdict(lambda: {"total_cost": 0, "file_count": 0})
    tasks = []

    with ThreadPoolExecutor() as executor:
        for model_name in tqdm(sorted(os.listdir("onnx")), disable=not logging):
            model_path = os.path.abspath(os.path.join("onnx", model_name))
            if os.path.isdir(model_path) or (filter and filter not in model_name) or not model_path.endswith(".onnx"):
                continue

            for file_index in range(DATAFILES_START, DATAFILES_START + training_files):
                tasks.append(
                    executor.submit(process_file, model_name, model_path, file_index, window_size, tinyphysicsmodel)
                )

        for future in tqdm(as_completed(tasks), total=len(tasks), disable=not logging):
            model_name, file_cost = future.result()
            model_costs[model_name]["total_cost"] += file_cost
            model_costs[model_name]["file_count"] += 1

    # Calculate average cost per model
    average_costs = [(model_name, values["total_cost"] / values["file_count"])
                     for model_name, values in model_costs.items()]

    # Sort models by average cost (ascending order)
    average_costs.sort(key=lambda x: x[1])

    # Print top best-performing models
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
