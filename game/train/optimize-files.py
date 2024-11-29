import os
import sys
import pandas as pd
from train import start_training
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directory containing all training files
simulations_dir = "./simulations/"
all_files = sorted(os.listdir(simulations_dir))

# Initialize a list to store results
results = []

def evaluate_single_file(file_name):
    """
    Train the model using a single file and return the validation score.
    """
    print(f"Evaluating file: {file_name}")

    # Run training with the selected file
    best_epoch, total_cost = start_training(
        epochs=10,  # Run for 6 epochs as requested
        analyze=False,
        logging=False,
        window_size=30,
        batch_size=44,
        lr=1.5e-5,
        seed=962,
        selected_files=[file_name]
    )

    return file_name, best_epoch, total_cost

if __name__ == "__main__":
    max_threads = 8  # You can adjust this based on your machine's CPU
    with ThreadPoolExecutor(max_threads) as executor:
        # Submit all files for evaluation
        future_to_file = {executor.submit(evaluate_single_file, file_name): file_name for file_name in all_files}

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                file_name, best_epoch, total_cost = future.result()
                # Append results to the list
                results.append({
                    "file_name": file_name,
                    "best_epoch": best_epoch,
                    "validation_score": total_cost
                })
            except Exception as e:
                print(f"Error while evaluating file {file_name}: {e}")

    # Sort results by validation score (ascending order)
    sorted_results = sorted(results, key=lambda x: x["validation_score"])

    # Print sorted results
    print("\n--- Results (Sorted by Validation Score) ---")
    for result in sorted_results:
        print(f"File: {result['file_name']}, Best Epoch: {result['best_epoch']}, Validation Score: {result['validation_score']:.4f}")

    # Save results to a CSV
    results_df = pd.DataFrame(sorted_results)
    results_df.to_csv("file_evaluation_results.csv", index=False)
