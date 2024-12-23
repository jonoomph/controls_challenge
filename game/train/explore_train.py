import collections
from threading import Lock
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


import torch
import torch.nn as nn
from tqdm import tqdm

import test_models

# Global lock for ONNX export
onnx_export_lock = Lock()


from explore_model import PIDControllerNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import string
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*variable length with LSTM.*")

import multiprocessing

# Get the total number of CPU cores
total_cores = multiprocessing.cpu_count()
test_threads = 4
leave_threads = 2

# Set the number of threads to use (total cores - 4)
num_threads = max(1, total_cores - test_threads - leave_threads)
torch.set_num_threads(num_threads)

print(f"Total Cores: {total_cores}")
print(f"Using {num_threads} threads for PyTorch.")
print(f"Using {test_threads} threads for Validation testing.")

from concurrent.futures import ThreadPoolExecutor
TESTING_EXECUTOR=ThreadPoolExecutor(max_workers=test_threads)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TrainingRun:
    def __init__(self, epoch, window_size, batch_size, optimizer, model, loss_fn, clamp_min, clamp_max):
        self.epoch = epoch
        self.total_loss = 0
        self.predictions = []
        self.solved = []
        self.steer_costs = []
        self.window_size = window_size
        self.input_window = []
        self.output_window = []
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.clamp_min=clamp_min
        self.clamp_max=clamp_max

    def process_batch(self, validate):
        if len(self.predictions) >= self.batch_size:
            # Shuffle the batch
            combined = list(zip(self.predictions, self.solved))
            random.shuffle(combined)
            shuffled_predictions, shuffled_solved = zip(*combined)

            # Stack the predictions, solved data, and costs
            predictions_tensor = torch.cat(shuffled_predictions).to(device)
            solved_tensor = torch.tensor(shuffled_solved, dtype=torch.float32).to(device)

            # Clamp solved_tensor to the range [-6, 6]
            solved_tensor = torch.clamp(solved_tensor, min=-4, max=4)

            # Normalize the clamped range to [0, 1]
            solved_tensor = (solved_tensor + 4) / 8

            # Apply the weights to the loss
            loss = self.loss_fn(predictions_tensor, solved_tensor)

            if not validate:
                # Backpropagation: clear the gradients, perform backpropagation, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Clear the lists for the next batch
            self.predictions.clear()
            self.solved.clear()
            self.steer_costs.clear()

            # Log the loss
            self.total_loss += loss.item()

    def train(self, state_input, torque_costs, validate):
        # Ensure state_input is a flattened 1D tensor
        state_input = state_input.squeeze(0)  # Remove batch dimension if present

        # Extract previous steer torque (15th index in the input state)
        previous_steer = state_input[14]  # The 15th value (index 14)

        # Extract the last 15 candidate steer torques
        candidate_torques = state_input[-15:]  # Last 15 values

        # Compute differences: candidate torques relative to the previous steer
        candidate_diffs = candidate_torques - previous_steer

        # Update state_input by replacing the last 15 values with the computed differences
        updated_state_input = torch.cat((state_input[:-15], candidate_diffs))

        # Store the updated input and output in their respective windows
        self.input_window.append(updated_state_input)
        self.output_window.append(torque_costs)

        # Process only if we have enough data for one window
        if len(self.input_window) >= self.window_size:
            # Stack the inputs to form a windowed tensor
            input_tensor = torch.stack(self.input_window[-self.window_size:]).unsqueeze(0).squeeze(2)

            # Get the model output
            model_output = self.model(input_tensor)

            # Store the prediction and the PID output
            self.predictions.append(model_output)
            self.solved.append(self.output_window[-1])

            # Process the batch (if needed)
            self.process_batch(validate)

            # Slide the window
            self.input_window.pop(0)
            self.output_window.pop(0)


def export_model(epoch=None, prefix="model", window_size=7, model=None, optimizer=None, logging=True):
    # Save the trained model to an ONNX file
    model_name = f"onnx/model-{prefix}-{epoch}.onnx"
    if logging:
        print(f"Exporting model: {model_name}")

    # Adjust the dummy input size according to the model's expected input size
    dummy_input = torch.randn(1, window_size, model.input_size, requires_grad=True).to(device)

    # Serialize ONNX export using a global lock
    with onnx_export_lock:
        try:
            # Export the model to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                model_name,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size', 1: 'window_size'}, 'output': {0: 'batch_size'}}
            )
        except Exception as e:
            print(f"Error during ONNX export for epoch {epoch}: {e}")
            raise

    # Export the model and optimizer states (PyTorch format)
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'onnx/model-{prefix}-{epoch}.pth')
    except Exception as e:
        print(f"Error saving PyTorch model for epoch {epoch}: {e}")
        raise


def plot_tensor_scores(file_path, candidate_torques_list, score_diffs_list, score_min=-6, score_max=6):
    """
    Plots filled candidate torques over time, colorized based on score diffs.

    Args:
        file_path (str): Path to the training file (used for plot title).
        candidate_torques_list (list of torch.Tensor): Tensors (9,) representing candidate torques at each timestep.
        score_diffs_list (list of torch.Tensor): Tensors (9,) representing simulation score differences.
        score_min (float): Minimum score for colormap scaling.
        score_max (float): Maximum score for colormap scaling.
    """
    num_candidates = len(candidate_torques_list[0])  # Number of candidates
    time_steps = torch.arange(len(candidate_torques_list))  # Time steps as tensor

    # Define a colormap: green for low scores, red for high scores
    def score_to_color(score, vmin=score_min, vmax=score_max):
        normalized_score = (score - vmin) / (vmax - vmin)  # Normalize score between 0 and 1
        normalized_score = min(max(normalized_score.item(), 0), 1)  # Ensure within [0, 1]
        return (1 - normalized_score, normalized_score, 0)  # RGB: Green -> Red

    plt.figure(figsize=(12, 6))

    # Plot each candidate with filled areas and colorized by score
    for i in range(num_candidates):
        # Extract torque values and score diffs for this candidate
        candidate_values = torch.tensor([candidates[i] for candidates in candidate_torques_list]).cpu()
        candidate_scores = torch.tensor([scores[i] for scores in score_diffs_list]).cpu()

        # Determine color for each timestep based on the score
        colors = [score_to_color(score) for score in candidate_scores]

        # Fill the area under each curve with the corresponding color
        for t in range(len(time_steps) - 1):
            plt.fill_between(
                [time_steps[t], time_steps[t + 1]],
                [candidate_values[t], candidate_values[t + 1]],
                color=to_rgba(colors[t], alpha=0.6)  # Add transparency
            )

    # Plot details
    plt.title(f"Candidate Torques Over Time: {file_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("Torque Values")
    plt.grid(True)

    # Add a color gradient legend for scores
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdYlGn_r", norm=Normalize(vmin=score_min, vmax=score_max))
    sm.set_array([])  # Dummy array for the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Candidate Scores")

    plt.show()


EXPORT_INTERVAL = 5
simulations_dir = "../data-explored/"


def start_training(epochs=65, window_size=7, logging=True, analyze=True, batch_size=36, lr=0.0001,
                   loss_fn=nn.MSELoss(), seed=2002, selected_files=None, hidden_size=80, clamp_min=-5, clamp_max=1):
    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model
    model = PIDControllerNet(window_size=window_size, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Initialize a dict to track loss per file across epochs
    file_loss_dict = collections.defaultdict(list)

    def update_file_loss(file_name, file_loss):
        # Add the current epoch's loss for the file to the dict
        file_loss_dict[file_name].append(file_loss)

    def analyze_worst_files(top_n=20):
        # Calculate the mean loss per file across all epochs
        mean_losses = {file: sum(losses) / len(losses) for file, losses in file_loss_dict.items()}

        # Sort files by their mean loss in descending order
        sorted_files = sorted(mean_losses.items(), key=lambda x: x[1], reverse=True)

        # Print the top worst files
        print(f"{'File':<20} {'Mean Loss':<15} {'Epoch Count':<10}")
        print("-" * 45)

        for i, (file, mean_loss) in enumerate(sorted_files[:top_n], 1):
            epoch_count = len(file_loss_dict[file])
            print(f"{i:<3} {file:<20} {mean_loss:<15.6f} {epoch_count:<10}")
        print("\nEnd of Top Worst Files Report")

    # Track the best epoch and its validation cost
    best_epoch = float('inf')
    best_cost = float('inf')

    # Define the callback function for when the testing completes
    def on_testing_complete(epoch, cost):
        """Callback function to log results to TensorBoard and track the best epoch."""
        nonlocal best_epoch, best_cost
        if logging:
            print(f"Epoch {epoch}, Validation Cost: {cost}")
            writer.add_scalar('Metrics/Validation Cost', cost, epoch)
        # Update the best epoch and cost if the current epoch is better
        if cost < best_cost:
            best_epoch = epoch
            best_cost = cost

    # Setup SummaryWriter for TensorBoard logging
    if logging:
        writer = SummaryWriter()

    threads = []

    # Get all data files in the simulations directory
    all_files = sorted(os.listdir(simulations_dir))

    # Filter files based on selected_files
    if selected_files is not None:
        datafiles = [f for f in all_files if f in selected_files]
    else:
        datafiles = all_files

    # Determine validation split index
    validation_split = 0.2
    split_idx = int(len(datafiles) * (1 - validation_split))
    print(f"Training on {split_idx} files, Validating on {len(datafiles) - split_idx} files.")

    for epoch in range(epochs):
        epoch_training_loss = 0
        epoch_validation_loss = 0

        DATAFILES = tqdm(datafiles, disable=not logging, position=0)
        for idx, filename in enumerate(DATAFILES):
            print(filename)
            validate = idx >= split_idx
            run = TrainingRun(epoch, window_size=window_size, batch_size=batch_size,
                              optimizer=optimizer, model=model, loss_fn=loss_fn, clamp_min=clamp_min, clamp_max=clamp_max)

            if filename.endswith('.pth'):
                file_path = os.path.join(simulations_dir, filename)

                # Load the tensor from the file
                tensor_data = torch.load(file_path)

                # Optional plot of tensor data and scores
                # plot_tensor_scores(file_path,
                #                    [torques[0].squeeze()[-15:] for torques in tensor_data],
                #                    [torch.tensor(torques[1], dtype=torch.float32).to(device) for torques in tensor_data])

                for row in tensor_data:
                    input_tensor = row[0].to(device)
                    torque_costs = row[1]
                    if input_tensor.shape[1] == 30:
                        run.train(input_tensor, torque_costs, validate=validate)

                # Accumulate loss
                if validate:
                    epoch_validation_loss += run.total_loss
                else:
                    epoch_training_loss += run.total_loss

                # Track loss per file
                update_file_loss(file_path, run.total_loss)

        # Log and calculate average losses
        avg_training_loss = epoch_training_loss / split_idx if split_idx > 0 else 0
        avg_validation_loss = epoch_validation_loss / (len(datafiles) - split_idx) if split_idx < len(datafiles) else 0

        if logging:
            writer.add_scalar('Metrics/Training Loss', avg_training_loss, epoch)
            writer.add_scalar('Metrics/Validation Loss', avg_validation_loss, epoch)

        # Export model
        export_model(epoch, prefix, window_size, model, optimizer, logging)

        # Start testing in a new thread with a callback
        def threaded_testing(epoch, callback):
            cost = test_models.start_testing(
                f"{prefix}-{epoch}",
                logging=False,
                window_size=window_size,
                executor=TESTING_EXECUTOR
            )
            callback(epoch, cost)  # Trigger callback with results

        # if analyze or epoch == epochs - 1:
        #     test_thread = threading.Thread(target=threaded_testing, args=(epoch, on_testing_complete))
        #     test_thread.start()
        #     threads.append(test_thread)

    # Ensure all threads complete before concluding training
    for t in threads:
        t.join()

    if logging:
        if analyze:
            print('\nAnalyze Models...')
            analyze_worst_files()
        print("\nTraining completed!")
        writer.close()

    # Return the best epoch and its validation cost
    return best_epoch, best_cost


if __name__ == "__main__":
    # Trial 2 finished with value: 48.776776926369166 and parameters: {'batch_size': 43, 'window_size': 34, 'hidden_size': 102}. Best is trial 2 with value: 48.776776926369166.
    best_epoch, best_cost = start_training(epochs=200, analyze=True, logging=True, lr=1.5e-4, seed=962,
                                           batch_size=10, window_size=30, hidden_size=80, clamp_min=-4, clamp_max=1)
    print(best_epoch, best_cost)
