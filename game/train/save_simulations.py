import math
import os
import random
from pathlib import Path

import torch

import tinyphysics
from controllers import replay

SIM = None
EPOCHS = 1
DATAFILES_START = 0
DATAFILES_LENGTH = 500
MAX_TRAINING_ROWS = 600
INCREMENT_TRAINING_ROWS_EVERY = 999
INCREMENT_TRAINING_ROWS_BY = 20
CONTEXT_LENGTH = 20
model_path = Path('../../models/tinyphysics.onnx')


class Controller:
    def __init__(self, epoch=0, internal_pid=None):
        self.epoch = epoch
        self.total_loss = 0
        self.internal_pid = internal_pid
        self.prev_state = None
        self.prev_actions = []
        self.cumulative_reward = 0  # Added for accumulating reward over a batch
        self.current_batch_states = []  # To store states over a batch
        self.current_batch_actions = []  # To store actions (PID values) over a batch
        self.stopped = False
        self.started = False
        self.pid_guess = None
        self.replay_buffer = []

    def store_transition(self, state, action):
        self.replay_buffer.append((state, action))

    def average(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        return v_ego_m_s / max_m_s

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        global SIM

        # Compute the differences from the current state for each segment
        future_segments = [(0, 8), (8, 16), (16, 24)]
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

        # Get action from controller
        action = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan)

        # Override initial steer commands
        if self.internal_pid.step_idx < 100:
            steer_command = SIM.data.get("steer_command")[self.internal_pid.step_idx]
            action = steer_command

        # Store transition in the replay buffer
        self.store_transition(state_input, action)
        self.prev_actions.append(action)

        return action


def safe_float(value):
    try:
        # Try converting the value to float
        return float(value)
    except ValueError:
        # If conversion fails (e.g., NaN or empty), return None
        return None


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


def start_training():
    global MAX_TRAINING_ROWS, SIM
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    # Get random files
    file_list = get_random_files('../data/', num_files=DATAFILES_LENGTH, seed=1979)

    for epoch in range(EPOCHS):
        for file_name in file_list:
            level_num = int(os.path.splitext(file_name)[0])
            data_path = os.path.join('../../data/', f'{level_num:05}.csv')

            for controller_name, internal_controller in [("PIDReplay", replay.Controller(level_num=level_num))]:
                # Create simulator
                SIM = tinyphysics.TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=Controller(epoch, internal_controller), debug=False)
                internal_controller.initial_steer = [steer for steer in SIM.data.get("steer_command") if not math.isnan(steer)]

                # Iterate through data rows (where steering was active)
                SIM.rollout()

                # Save model data
                cost = SIM.compute_cost()
                model_data = SIM.controller.replay_buffer
                print(f'Saving Model: {level_num:05d}', cost["total_cost"])
                torch.save(model_data, f'simulations/{level_num:05d}.pth')

    print("Training completed!")


if __name__ == "__main__":
    start_training()
