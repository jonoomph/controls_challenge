import optuna
from train import start_training
import os
import sys

# Add parent directory to sys.path to locate modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(parent_dir)

# Directory containing all training files
simulations_dir = "./simulations/"
all_files = sorted(os.listdir(simulations_dir))  # Fetch all files in the directory


# Define the objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters like learning rate, window size, and model size.
    """

    # Suggest hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    window_size = trial.suggest_int("window_size", 10, 60)
    hidden_size = trial.suggest_int("hidden_size", 75, 105)
    steer_clamp_min = trial.suggest_float("steer_clamp_min", -8, -0.5)
    steer_clamp_max = trial.suggest_float("steer_clamp_max", 0.5, 8)

    # Run training with the suggested hyperparameters
    best_epoch, total_cost = start_training(
        epochs=30,
        analyze=True,
        logging=False,
        window_size=window_size,
        batch_size=44,
        lr=lr,
        seed=962,
        hidden_size=hidden_size,
        clamp_min=steer_clamp_min,
        clamp_max=steer_clamp_max,
    )

    # Log hyperparameters and best epoch as user attributes for this trial
    trial.set_user_attr("best_epoch", best_epoch)

    return total_cost


# Main code for running the Optuna study
if __name__ == "__main__":
    # Create a study object
    study = optuna.create_study(direction='minimize')  # We want to minimize the validation cost

    # Optimize the objective function for a number of trials
    study.optimize(objective, n_trials=60, n_jobs=-1, show_progress_bar=True)

    # Print the best trial's result
    print("\n--- Best Trial ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best trial parameters: {study.best_trial.params}")
    print(f"Best validation score (loss): {study.best_trial.value}")
    print(f"Best epoch: {study.best_trial.user_attrs['best_epoch']}")

    # Save all trial results to a CSV
    study.trials_dataframe().to_csv("optuna_results.csv")

    # Loop through all trials
    print("\n--- All Trials ---")
    for trial in study.trials:
        print(f"Trial #{trial.number}")
        print(f"  Best Epoch: {trial.user_attrs.get('best_epoch', 'N/A')}")
        print(f"  Validation Score (Loss): {trial.value}")
