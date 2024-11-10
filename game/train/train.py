import torch
import torch.nn as nn
from tqdm import tqdm
import test_models
import threading
import numpy as np


from model import PIDControllerNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import string
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*variable length with LSTM.*")
import collections


class TrainingRun:
    def __init__(self, epoch, window_size, batch_size, optimizer, model, loss_fn):
        self.epoch = epoch
        self.total_loss = 0
        self.predictions = []
        self.solved = []
        self.window_size = window_size
        self.input_window = []
        self.output_window = []
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn

    def process_batch(self):
        if len(self.predictions) >= self.batch_size:
            # Shuffle the batch
            combined = list(zip(self.predictions, self.solved))
            random.shuffle(combined)
            shuffled_predictions, shuffled_solved = zip(*combined)

            # Stack the predictions and solved data
            predictions_tensor = torch.cat(shuffled_predictions)
            solved_tensor = torch.tensor(shuffled_solved, dtype=torch.float32).unsqueeze(1)

            # Calculate loss
            loss = self.loss_fn(predictions_tensor, solved_tensor)

            # Backpropagation: clear the gradients, perform backpropagation, and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clear the lists for the next batch
            self.predictions.clear()
            self.solved.clear()

            # Log the loss
            self.total_loss += loss.item()

    def train(self, state_input, steer_torque, diff_threshold=0.1):
        # Store the current input and output in their respective windows
        self.input_window.append(state_input)
        self.output_window.append(steer_torque)

        # Process only if we have enough data for one window
        if len(self.input_window) >= self.window_size:

            # # Stack the inputs to form a windowed tensor
            # window_tensor = torch.stack(self.input_window[-self.window_size:])
            # # Extract the lateral accel diff from segment 0 (closest to current)
            # avg_diff = torch.mean(torch.abs(window_tensor[:, 0, 0])).item()
            # # # Skip this window if the average difference exceeds the threshold
            # if avg_diff > diff_threshold:
            #     # Slide the window without processing
            #     self.input_window.pop(0)
            #     self.output_window.pop(0)
            #     return

            # Stack the inputs to form a windowed tensor
            input_tensor = torch.stack(self.input_window[-self.window_size:]).unsqueeze(0).squeeze(2)

            # Get the model output
            model_output = self.model(input_tensor)

            # Store the prediction and the PID output
            self.predictions.append(model_output)
            self.solved.append(self.output_window[-1])

            # Process the batch (if needed)
            self.process_batch()

            # Slide the window
            self.input_window.pop(0)
            self.output_window.pop(0)


def export_model(epoch=None, prefix="model", window_size=7, model=None, optimizer=None, logging=True):
    # Save the trained model to an ONNX file
    model_name = f"onnx/model-{prefix}-{epoch}.onnx"
    if logging:
        print(f"Exporting model: {model_name}")

    # Adjust the dummy input size according to the model's expected input size
    dummy_input = torch.randn(1, window_size, model.input_size, requires_grad=True)

    # Export the model for onnx
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

    # Export model and optimizer (in pytorch syntax)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'onnx/model-{prefix}-{epoch}.pth')


EXPORT_INTERVAL = 5
simulations_dir = "./simulations/"


def start_training(epochs=65, window_size=7, logging=True, analyze=True, batch_size=36, lr=0.0001, loss_fn=nn.MSELoss(), seed=2002):
    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model
    model = PIDControllerNet(window_size=window_size)
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

    # Define the callback function for when the testing completes
    def on_testing_complete(epoch, cost):
        """Callback function to log results to TensorBoard."""
        if logging:
            writer.add_scalar('Metrics/Validation Cost', cost, epoch)

    # Setup SummaryWriter for TensorBoard logging
    if logging:
        writer = SummaryWriter()

    total_loss = 0
    threads = []

    # forward = np.linspace(0.03, 1.0, round(epochs / 2.0), endpoint=False)
    # backward = np.linspace(1.0, 0.03, round(epochs / 2.0))
    # diff_thresholds = np.concatenate((forward, backward))
    diff_thresholds = np.linspace(1.0, 0.04, epochs)

    # curve for diff
    # x_sigmoid = np.linspace(1.0, 0.03, epochs)
    # sigmoid_curve = 1 / (1 + np.exp(-x_sigmoid))
    # diff_thresholds = 1.0 + (0.03 - 1.0) * sigmoid_curve
    # reversed(diff_thresholds)

    for epoch in range(epochs):
        epoch_loss = 0

        # Loop through each file in the simulations directory
        DATAFILES = tqdm(sorted(os.listdir(simulations_dir)), disable=not logging)
        for filename in DATAFILES:
            run = TrainingRun(epoch, window_size=window_size, batch_size=batch_size,
                              optimizer=optimizer, model=model, loss_fn=loss_fn)

            if filename.endswith('.pth'):
                file_path = os.path.join(simulations_dir, filename)

                # Load the tensor from the file
                tensor_data = torch.load(file_path)

                # Get the subset of data
                tensor_data_subset = tensor_data[80:]

                for row in tensor_data_subset:
                    input_tensor = row[0]
                    steer_torque = row[1]
                    run.train(input_tensor, steer_torque)

                # Add loss
                epoch_loss += run.total_loss

                # Track loss per file
                update_file_loss(file_path, run.total_loss)

        # Track all epochs
        total_loss += epoch_loss / len(DATAFILES)

        # Export model
        export_model(epoch + 1, prefix, window_size, model, optimizer, logging)

        # Log to graph
        if logging:
            writer.add_scalar('Metrics/Training Loss', epoch_loss / len(DATAFILES), epoch)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(DATAFILES)}")

            # Start testing in a new thread with a callback
            def threaded_testing(epoch, callback):
                cost = test_models.start_testing(
                    f"{prefix}-{epoch + 1}",
                    logging=logging,
                    window_size=window_size,
                    training_files=20
                )
                callback(epoch, cost)  # Trigger callback with results

            test_thread = threading.Thread(target=threaded_testing, args=(epoch, on_testing_complete))
            test_thread.start()
            threads.append(test_thread)

    # Ensure all threads complete before concluding training
    for t in threads:
        t.join()

    if logging:
        print("\nTraining completed!")
        writer.close()

    if analyze:
        if logging:
            print('\nAnalyze Models...')
            analyze_worst_files()
        # Return simulation cost: filter= f"{prefix}-{epochs}"
        total_cost = test_models.start_testing(f"{prefix}-{epochs}", logging=logging, window_size=window_size, training_files=20)
        return total_cost
    else:
        # Return average training loss
        return total_loss / epochs

if __name__ == "__main__":
    # Trial 88: {'lr': 8.640162515565103e-05, 'batch_size': 44, 'window_size': 22}
    loss = start_training(epochs=150, analyze=True, logging=True, window_size=30, batch_size=44, lr=0.00001, seed=962)
    print(loss)

