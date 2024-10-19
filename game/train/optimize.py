import optuna
import torch.nn as nn
from train import start_training


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for tuning
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    window_size = trial.suggest_int('window_size', 3, 20)

    # Define loss function (MSE in this case, could be something else)
    loss_fn = nn.MSELoss()

    # Call your training function with the suggested hyperparameters
    average_loss = start_training(
        epochs=50,
        window_size=window_size,
        logging=False,
        analyze=True,
        batch_size=batch_size,
        lr=lr,
        loss_fn=loss_fn
    )

    # Return the final average loss from the training function
    return average_loss

# Main code for running the Optuna study
if __name__ == "__main__":
    # Create a study object
    study = optuna.create_study(direction='minimize')  # We want to minimize the loss

    # Optimize the objective function for a number of trials
    study.optimize(objective, n_trials=20, n_jobs=6, show_progress_bar=True)

    # Print the best trial's result
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best value (loss): {study.best_trial.value}")

    # Optionally, save the study results
    study.trials_dataframe().to_csv("optuna_results.csv")
