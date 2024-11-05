import math
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

import string
import tinyphysics
from model import PIDControllerNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*variable length with LSTM.*")

SIM = None
DATAFILES_START = 0
model_path = Path('../../models/tinyphysics.onnx')


class Controller:
    def __init__(self, epoch=0, model=None, optimizer=None, window_size=22):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.prev_actions = []
        self.replay_buffer = []
        self.window_size = window_size
        self.input_window = []

    def store_transition(self, state, action):
        self.replay_buffer.append((state, action))

    def average(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        return v_ego_m_s / max_m_s

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
        previous_action = 0
        if len(self.prev_actions) > 0:
            previous_action = self.prev_actions[-1]

        # Flatten the differences into a single list
        state_input_list = (diff_values['lataccel'] + diff_values['roll'] + diff_values['v_ego'] + diff_values['a_ego'] + [previous_action])

        # Prepare the state as input for the model
        state_input = torch.tensor(state_input_list, dtype=torch.float32).unsqueeze(0)

        # Update time-series window
        self.input_window.append(state_input)

        # Run model if window is ready
        action = 0
        if len(self.input_window) >= self.window_size: # and self.step_idx >= 93:
            # Stack the inputs to form a windowed tensor
            input_tensor = torch.stack(self.input_window[-self.window_size:]).unsqueeze(0).squeeze(2)

            # Get the model output
            with torch.no_grad():
                action = self.model(input_tensor).item()

        # Override initial steer commands
        if not math.isnan(steer):
            action = steer
        else:
            # Store transition in the replay buffer
            self.store_transition(state_input, action)

        self.prev_actions.append(action)
        return action


def get_random_files(folder_path, num_files=1, seed=1):
    # Set the seed for repeatability
    random.seed(seed)

    # Get a list of all files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".npy")]

    # If there are fewer files than the requested number, return all files
    if len(all_files) <= num_files:
        return all_files

    # Randomly sample the files
    sampled_files = random.sample(all_files, num_files)

    return sampled_files


def start_finetune(model_name="", epochs=65, window_size=7, logging=True, analyze=True, batch_size=36, lr=0.0001, loss_fn=nn.MSELoss(), seed=2002, num_files=99):
    global SIM
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    prefix = ''.join(random.choice(string.ascii_letters) for _ in range(5))
    print(f"Start fine tuning job: {prefix}")

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model
    model = PIDControllerNet(window_size=window_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the model checkpoint
    checkpoint = torch.load(model_name)

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Setup SummaryWriter for TensorBoard logging
    if logging:
        writer = SummaryWriter()

    # Get random files
    total_loss = 0

    file_list = sorted(get_random_files('../data/', num_files=num_files, seed=1979))
    for epoch in range(epochs):
        epoch_loss = 0
        for file_name in tqdm(file_list, disable=not logging):
            level_num = int(os.path.splitext(file_name)[0])
            data_path = os.path.join('../../data/', f'{level_num:05}.csv')

            # Create simulator
            SIM = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=Controller(epoch, model, optimizer, window_size), debug=False)

            # Iterate through data rows (where steering was active)
            SIM.rollout()

            # Save model data
            epoch_loss += SIM.compute_cost()["total_cost"]

        # Log to graph
        if logging:
            writer.add_scalar('Metrics/Training Loss', epoch_loss / len(file_list), epoch)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss / len(file_list)}")

        # Track all epochs
        total_loss += epoch_loss / len(file_list)


    if logging:
        print("\nTraining completed!")
        writer.close()

    print("Training completed!")


if __name__ == "__main__":
    loss = start_finetune(model_name="onnx/model-OKBCV-60.pth", epochs=80, analyze=True, logging=True, window_size=30, batch_size=44, lr=0.00004, seed=962)
    print(loss)
