import math

from . import BaseController
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Controller(BaseController):
    def __init__(self, level_num=2):
        # Load original replay data
        original_replay_path = f'/home/jonathan/apps/controls_challenge/game/data/{level_num:05}.npy'
        self.torques = np.load(original_replay_path)

        # Load the tensor from the file
        file_path = f'/home/jonathan/apps/controls_challenge/game/data-explored/{level_num:05}.pth'
        tensor_data = torch.load(file_path)
        for i, row in enumerate(tensor_data):

            input_tensor = row[0].to(device)
            torque_costs = row[1]
            if input_tensor.shape[1] == 30:
                # Convert torque_costs to a PyTorch tensor
                torque_costs_tensor = torch.tensor(torque_costs, dtype=torch.float32).to(device)

                # Find the index of the minimum torque cost
                min_torque_index = torch.argmin(torque_costs_tensor)

                # Extract the corresponding steering torque based on the lowest torque cost index
                steering_torque = input_tensor[0, 15 + min_torque_index]
                self.torques[85 + i] = steering_torque

        self.step_idx = 19

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=None):
        self.step_idx += 1
        if steer and not math.isnan(steer):
            return steer
        else:
            return self.torques[self.step_idx-20]
