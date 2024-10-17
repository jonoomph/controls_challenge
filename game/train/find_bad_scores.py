import os
from pathlib import Path
from tqdm import tqdm
import tinyphysics
from controllers.pid_model import Controller

DATAFILES_START = 0
DATAFILES_LENGTH = 500
MAX_TRAINING_ROWS = 600
model_path = Path('../../models/tinyphysics.onnx')


def find_bad_scores():
    # Use the default pid_model (no argument needed for Controller)
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(str(model_path), debug=False)

    # List to store the scores for each file
    scores = []

    for file_index in tqdm(range(DATAFILES_START, DATAFILES_START + DATAFILES_LENGTH)):
        data_path = f'../../data/{file_index:05d}.csv'

        # Create the controller and simulator
        controller = Controller()  # No model path passed, use default pid_model
        sim = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

        # Run the simulation and compute the cost
        sim.rollout()
        cost = sim.compute_cost()

        # Store the result (file index and total cost)
        scores.append((file_index, cost["total_cost"]))

    # Sort the scores by total_cost in descending order (worst scores first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Print the results in a nice format
    print(f"Top {DATAFILES_LENGTH} Simulations Sorted by Total Cost (Worst to Best):")
    for rank, (file_index, total_cost) in enumerate(scores, 1):
        print(f"#{rank} | File: {file_index:05d}.csv | Total Cost: {total_cost:.4f}")


if __name__ == "__main__":
    find_bad_scores()
