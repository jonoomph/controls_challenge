import collections
import threading
from threading import Lock

import torch
import torch.nn as nn
from tqdm import tqdm

import test_models

# Global lock for ONNX export
onnx_export_lock = Lock()


from model import PIDControllerNet
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
        self.cost_window = []
        self.cost_total_window = []
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.clamp_min=clamp_min
        self.clamp_max=clamp_max

    def process_batch(self):
        if len(self.predictions) >= self.batch_size:
            # Shuffle the batch
            combined = list(zip(self.predictions, self.solved, self.steer_costs))
            random.shuffle(combined)
            shuffled_predictions, shuffled_solved, shuffled_costs = zip(*combined)

            # Stack the predictions, solved data, and costs
            predictions_tensor = torch.cat(shuffled_predictions).to(device)
            solved_tensor = torch.tensor(shuffled_solved, dtype=torch.float32).to(device)
            steer_cost_tensor = torch.tensor(shuffled_costs, dtype=torch.float32).unsqueeze(1).to(device)

            # Clamp steer costs to the specified range
            steer_cost_tensor = torch.clamp(steer_cost_tensor, min=self.clamp_min, max=self.clamp_max)

            # Linearly flip weights so -3.0 → 1.0, 1.0 → 0.0
            weight_tensor = 1.0 - (steer_cost_tensor - self.clamp_min) / (self.clamp_max - self.clamp_min)

            # Apply the weights to the loss
            weighted_loss = (self.loss_fn(predictions_tensor, solved_tensor) * weight_tensor).mean()

            # Backpropagation: clear the gradients, perform backpropagation, and update the weights
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()

            # Clear the lists for the next batch
            self.predictions.clear()
            self.solved.clear()
            self.steer_costs.clear()

            # Log the loss
            self.total_loss += weighted_loss.item()

    def train(self, state_input, steer_torques, steer_cost_diff, steer_cost_total):

        # Store the current input and output in their respective windows
        self.input_window.append(state_input)
        self.output_window.append(steer_torques)
        self.cost_window.append(steer_cost_diff)
        self.cost_total_window.append(steer_cost_total)

        # Process only if we have enough data for one window
        if len(self.input_window) >= self.window_size:
            # Stack the inputs to form a windowed tensor
            input_tensor = torch.stack(self.input_window[-self.window_size:]).unsqueeze(0).squeeze(2)

            # Get the model output
            model_output = self.model(input_tensor)

            # Store the prediction and the PID output
            self.predictions.append(model_output)
            self.solved.append(self.output_window[-1][0:2])
            self.steer_costs.append(self.cost_window[-1])

            # Process the batch (if needed)
            self.process_batch()

            # Slide the window
            self.input_window.pop(0)
            self.output_window.pop(0)
            self.cost_window.pop(0)
            self.cost_total_window.pop(0)


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


EXPORT_INTERVAL = 5
simulations_dir = "./simulations/"


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

    total_loss = 0
    threads = []

    for epoch in range(epochs):
        epoch_loss = 0

        # Get all data files in the simulations directory
        all_files = sorted(os.listdir(simulations_dir))

        # Filter files based on selected_files
        if selected_files is not None:
            datafiles = [f for f in all_files if f in selected_files]
        else:
            datafiles = all_files

        DATAFILES = tqdm(datafiles, disable=not logging, position=0)
        for filename in DATAFILES:
            run = TrainingRun(epoch, window_size=window_size, batch_size=batch_size,
                              optimizer=optimizer, model=model, loss_fn=loss_fn, clamp_min=clamp_min, clamp_max=clamp_max)

            if filename.endswith('.pth'):
                file_path = os.path.join(simulations_dir, filename)

                # Load the tensor from the file
                tensor_data = torch.load(file_path)

                # Get the subset of data
                tensor_data_subset = tensor_data[80:-80]

                for row in tensor_data_subset:
                    input_tensor = row[0].to(device)
                    steer_torques = row[1]
                    steer_cost_diff = row[2]
                    steer_cost_total = row[3]
                    run.train(input_tensor, steer_torques, steer_cost_diff, steer_cost_total)

                # Add loss
                epoch_loss += run.total_loss

                # Track loss per file
                update_file_loss(file_path, run.total_loss)

        # Track all epochs
        total_loss += epoch_loss / len(DATAFILES)

        # Export model
        export_model(epoch, prefix, window_size, model, optimizer, logging)

        # Log to graph
        if logging:
            writer.add_scalar('Metrics/Training Loss', epoch_loss / len(DATAFILES), epoch)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(DATAFILES)}")

        # Start testing in a new thread with a callback
        def threaded_testing(epoch, callback):
            cost = test_models.start_testing(
                f"{prefix}-{epoch}",
                logging=False,
                window_size=window_size,
                executor=TESTING_EXECUTOR
            )
            callback(epoch, cost)  # Trigger callback with results

        if analyze or epoch == epochs - 1:
            test_thread = threading.Thread(target=threaded_testing, args=(epoch, on_testing_complete))
            test_thread.start()
            threads.append(test_thread)

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
    best_epoch, best_cost = start_training(epochs=35, analyze=True, logging=True, lr=1.4e-5, seed=962,
                                           batch_size=44, window_size=30, hidden_size=80, clamp_min=-4, clamp_max=1)
    print(best_epoch, best_cost)
