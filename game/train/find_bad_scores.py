from pathlib import Path
from tqdm import tqdm
import tinyphysics
from controllers.pid_model import Controller
from concurrent.futures import ProcessPoolExecutor
from functools import partial

DATAFILES_START = 0
DATAFILES_LENGTH = 100
MAX_TRAINING_ROWS = 600
model_path = Path('../../models/tinyphysics.onnx')


def run_simulation(file_index, model_path):
    # Create the controller and simulator
    controller = Controller()  # No model path passed, use default pid_model
    data_path = f'../../data/{file_index:05d}.csv'
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(str(model_path), debug=False)
    sim = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

    # Run the simulation and compute the cost
    sim.rollout()
    cost = sim.compute_cost()

    # Return the result (file index and total cost)
    return (file_index, cost["total_cost"])


def find_bad_scores():
    # Prepare a partial function with the model path
    simulate_partial = partial(run_simulation, model_path=model_path)

    # Run simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(simulate_partial, range(DATAFILES_START, DATAFILES_START + DATAFILES_LENGTH)),
                            total=DATAFILES_LENGTH))

    # Sort the scores by total_cost in descending order (worst scores first)
    scores = sorted(results, key=lambda x: x[1], reverse=True)

    # Print the results in a nice format
    print(f"Top {DATAFILES_LENGTH} Simulations Sorted by Total Cost (Worst to Best):")
    for rank, (file_index, total_cost) in enumerate(scores, 1):
        print(f"#{rank} | File: {file_index:05d}.csv | Total Cost: {total_cost:.4f}")


if __name__ == "__main__":
    find_bad_scores()
