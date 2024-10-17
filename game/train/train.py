import torch
import torch.nn as nn
from tqdm import tqdm
import test_models

from model import PIDControllerNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random


EPOCHS = 80
WINDOW_SIZE = 7
EXPORT_INTERVAL = 5
BATCH_SIZE = 36

model = PIDControllerNet()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()


class Controller:
    def __init__(self, epoch, window_size):
        self.epoch = epoch
        self.total_loss = 0
        self.predictions = []
        self.solved = []
        self.window_size = window_size
        self.input_window = []
        self.output_window = []
        self.process_count = 0

    def process_batch(self):
        if len(self.predictions) >= BATCH_SIZE:
            # Shuffle the batch
            combined = list(zip(self.predictions, self.solved))
            random.shuffle(combined)
            shuffled_predictions, shuffled_solved = zip(*combined)

            # Stack the predictions and solved data
            predictions_tensor = torch.cat(shuffled_predictions)
            solved_tensor = torch.tensor(shuffled_solved, dtype=torch.float32).unsqueeze(1)

            # Calculate loss
            loss = loss_fn(predictions_tensor, solved_tensor)

            # Backpropagation: clear the gradients, perform backpropagation, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear the lists for the next batch
            self.predictions.clear()
            self.solved.clear()

            # Log the loss
            self.total_loss += loss.item()

    def train(self, state_input, steer_torque, diff_threshold=0.065):
        # Store the current input and output in their respective windows
        self.input_window.append(state_input)
        self.output_window.append(steer_torque)

        # Process only if we have enough data for one window
        if len(self.input_window) >= self.window_size:
            # Stack the inputs to form a windowed tensor
            window_tensor = torch.stack(self.input_window[-self.window_size:])

            # Extract the lateral accel diff from segment 0 (closest to current)
            avg_diff = torch.mean(torch.abs(window_tensor[:, 0, 0])).item()

            # # Skip this window if the average difference exceeds the threshold
            # if avg_diff > diff_threshold:
            #     # Slide the window without processing
            #     self.input_window.pop(0)
            #     self.output_window.pop(0)
            #     return
            # else:
            self.process_count += 1

            # Stack the inputs to form a windowed tensor
            input_tensor = torch.stack(self.input_window[-self.window_size:]).unsqueeze(0).squeeze(2)

            # Get the model output
            model_output = model(input_tensor)

            # Store the prediction and the PID output
            self.predictions.append(model_output)
            self.solved.append(self.output_window[-1])

            # Process the batch (if needed)
            self.process_batch()

            # Slide the window
            self.input_window.pop(0)
            self.output_window.pop(0)



def export_model(epoch=None):
    # Save the trained model to an ONNX file
    if epoch:
        model_name = f"onnx/lat_accel_predictor-{epoch}.onnx"
    else:
        model_name = f"onnx/lat_accel_predictor.onnx"
    print(f"Exporting model: {model_name}")

    # Adjust the dummy input size according to the model's expected input size (13 features)
    dummy_input = torch.randn(1, WINDOW_SIZE, 13, requires_grad=True)

    # Export the model
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



simulations_dir = "./simulations/"


def start_training():
    # Setup SummaryWriter for TensorBoard logging
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        epoch_loss = 0

        # Loop through each file in the simulations directory
        DATAFILES = tqdm(sorted(os.listdir(simulations_dir)))
        for filename in DATAFILES:
            controller = Controller(epoch, window_size=WINDOW_SIZE)

            if filename.endswith('.pth'):
                file_path = os.path.join(simulations_dir, filename)

                # Load the tensor from the file
                tensor_data = torch.load(file_path)

                # Get the subset of data
                tensor_data_subset = tensor_data #[80:]

                for row in tensor_data_subset:
                    input_tensor = row[0]
                    input_tensor.requires_grad_(True)
                    steer_torque = row[1]
                    controller.train(input_tensor, steer_torque)

                # Add loss
                epoch_loss += controller.total_loss

        # Log to graph
        writer.add_scalar('Average Loss', epoch_loss / len(DATAFILES), epoch)
        print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(DATAFILES)}")

        if (epoch + 1) % EXPORT_INTERVAL == 0 and (epoch + 1) > 15:
            export_model(epoch + 1)

    print('\nAnalyze Models...')
    test_models.start_testing()

    print("\nTraining completed!")
    writer.close()

if __name__ == "__main__":
    start_training()

