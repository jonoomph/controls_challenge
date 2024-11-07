import math
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

import string
import tinyphysics
from tinyphysics import DEL_T, LAT_ACCEL_COST_MULTIPLIER
from model import PIDControllerNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings
torch.autograd.set_detect_anomaly(False)

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*variable length with LSTM.*")

SIM = None
DATAFILES_START = 0
model_path = Path('../../models/tinyphysics.onnx')

import numpy as np
min_torque = -2.5
max_torque = 2.5
num_steps = 256
torque_levels = np.linspace(min_torque, max_torque, num_steps)

class Controller:
    def __init__(self, epoch=0, model=None, optimizer=None, window_size=30, batch_size=32, gamma=0.99):
        self.epoch = epoch
        self.model = model  # Pre-trained model
        self.optimizer = optimizer  # Optimizer for the model
        self.batch_size = batch_size  # Size of mini-batch for processing
        self.prev_actions = []
        self.input_window = []
        self.window_size = window_size
        self.predictions = []  # To store model predictions (future diffs)
        self.targets = []  # To store target values (zeros for minimizing future diffs)
        self.total_loss = 0
        self.loss_fn = nn.MSELoss()  # Loss function to minimize future diffs
        self.pending_model_output = None

    def store_transition(self, prediction, target_diff):
        # Store prediction and target diff in the current batch
        self.predictions.append(prediction)
        self.targets.append(target_diff)

    def process_batch(self):
        if len(self.predictions) >= self.batch_size and self.epoch > 0:
            # Shuffle the batch
            combined = list(zip(self.predictions, self.targets))
            random.shuffle(combined)
            shuffled_predictions, shuffled_solved = zip(*combined)

            # Stack the predictions and solved data
            predictions_tensor = torch.cat(shuffled_predictions)
            solved_tensor = torch.cat(shuffled_solved)

            # Calculate loss
            loss = self.loss_fn(predictions_tensor, solved_tensor)

            # Backpropagation: clear the gradients, perform backpropagation, and update the weights
            loss.backward()

            # Clear the lists for the next batch
            self.predictions.clear()
            self.targets.clear()

            # Log the loss
            self.total_loss += loss.item()

    def predict_best_torque(self, input_tensor, action, tolerance=0.01, large_adjust_threshold=0.5):
        # Clone the action to avoid in-place modification
        best_action = action.clone()

        # Initial torque and index calculation
        action_torque = best_action[0][0]
        torque_index = min(range(len(torque_levels)), key=lambda i: abs(torque_levels[i] - action_torque))

        # Calculate mean difference (this would likely be computed based on the model's future path prediction)
        mean_diff = torch.mean(best_action[0][1:]).item()

        # Determine adjustment direction based on mean_diff
        if abs(mean_diff) < tolerance or self.epoch == 0:
            # If mean_diff is within the tolerance, return the original torque (no adjustment needed)
            #print(f"Mean diff is within tolerance ({tolerance}), using original torque.")
            return action_torque

        # Decide adjustment step based on how far mean_diff is from zero
        adjustment_step = 1 #2 if abs(mean_diff) > large_adjust_threshold else 1

        # Increment or decrement the index within bounds
        if mean_diff < 0.0:
            # Increase torque if mean_diff is positive
            #new_torque_index = min(torque_index + adjustment_step, len(torque_levels) - 1)
            new_torque = action_torque + (2.5 / len(torque_levels))
        else:
            # Decrease torque if mean_diff is negative
            #new_torque_index = max(torque_index - adjustment_step, 0)
            new_torque = action_torque - (2.5 / len(torque_levels))

        #print(f"Original torque: {action_torque}, Adjusted torque: {new_torque_value} (mean_diff: {mean_diff})")
        return new_torque

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        # Compute the differences from the current state for each segment
        future_segments = [(0, 1), (1, 3), (2, 5), (5, 9), (9, 14), (14, 20)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(state.v_ego) - self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [state.a_ego - self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }

        # Previous steering torque
        previous_action = self.prev_actions[-1] if self.prev_actions else 0

        # Store transition (if pending ones)
        if self.pending_model_output is not None:
            # Prepare final best output
            best_output = torch.tensor([previous_action] + diff_values['lataccel'], dtype=torch.float32).unsqueeze(0)
            if abs(torch.mean(best_output[0][1:]).item()) < abs(torch.mean(self.pending_model_output[0][1:]).item()):
                self.store_transition(self.pending_model_output, best_output.detach())
            self.pending_model_output = None

        # Prepare the state as input for the model
        state_input_list = (
            diff_values['lataccel'] + diff_values['roll'] + diff_values['v_ego'] + diff_values['a_ego'] + [previous_action]
        )
        state_input = torch.tensor(state_input_list, dtype=torch.float32).unsqueeze(0)
        self.input_window.append(state_input.clone())

        if len(self.input_window) > self.window_size:
            self.input_window.pop(0)

        action = 0.0
        if len(self.input_window) >= self.window_size:
            # Stack and prepare the input tensor for the model
            input_tensor = torch.stack(self.input_window).unsqueeze(0).squeeze(2)  # Shape: (1, window_size, input_size)
            model_output = self.model(input_tensor)
            best_torque = self.predict_best_torque(input_tensor, model_output)

            # Store the prediction and target diff for batch processing (when in control of sim)
            if math.isnan(steer) and best_torque != model_output[:, 0].item():
                self.pending_model_output = model_output.clone()

            # Update steering action for the simulator
            action = best_torque.item()

        # Override initial steer values
        if not math.isnan(steer):
            action = steer
        else:
            # Process the batch if it's ready
            self.process_batch()

        self.prev_actions.append(action)

        return action

    def average(self, values):
        return sum(values) / len(values) if len(values) > 0 else 0

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        return v_ego_m_s / max_m_s


def get_random_files(folder_path, num_files=1, seed=1):
    random.seed(seed)
    all_files = [f for f in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".npy")]
    return all_files if len(all_files) <= num_files else random.sample(all_files, num_files)


def start_finetune(model_name="", epochs=5, window_size=30, logging=True, analyze=True, batch_size=36, lr=1e-6,
                   seed=2002, num_files=20):
    global SIM
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))
    print(f"Start fine tuning job: {prefix}")

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model
    model = PIDControllerNet(window_size=window_size, input_size=25)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the model checkpoint
    checkpoint = torch.load(model_name)

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #model.eval()

    # Setup SummaryWriter for TensorBoard logging
    if logging:
        writer = SummaryWriter()

    file_list = sorted(get_random_files('../data/', num_files=num_files, seed=1979))
    for epoch in range(epochs):
        epoch_loss = 0
        for file_name in tqdm(file_list, disable=not logging):
            optimizer.zero_grad()
            level_num = int(os.path.splitext(file_name)[0])
            data_path = os.path.join('../../data/', f'{level_num:05}.csv')

            controller = Controller(epoch, model, optimizer, window_size, batch_size=batch_size)
            SIM = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)
            SIM.rollout()

            # Save model data
            epoch_loss += SIM.compute_cost()["total_cost"]

            # Backpropagation: clear the gradients, perform backpropagation, and update the weights
            optimizer.step()

        # Log to graph
        if logging:
            writer.add_scalar('Metrics/Training Loss', epoch_loss / len(file_list), epoch)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(file_list)}")

    if logging:
        print("\nTraining completed!")
        writer.close()

    # Save the fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'fine_tuned_model_{prefix}.pth')


if __name__ == "__main__":
    loss = start_finetune(model_name="onnx/model-DOxSG-25.pth", epochs=10, analyze=True, logging=True, window_size=30, batch_size=44, lr=1e-6, seed=962)
    print(loss)
