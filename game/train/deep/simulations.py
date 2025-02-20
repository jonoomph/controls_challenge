# simulations.py
import os
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import tinyphysics
from controllers import pid_model
import config


# DualController: uses the supervised PID model or the Q-network via epsilon-greedy.
class DualController:
    def __init__(self, pid_model, q_model, config, current_epoch=0):
        self.pid_model = pid_model      # Teacher model
        self.q_model = q_model          # Q-network (can be None initially)
        self.config = config
        self.current_epoch = current_epoch
        self.prev_actions = []
        # Each transition: (state, action, reward, next_state, done)
        self.replay_buffer = []

        # Generate OU noise with a randomized sigma between 0.0 and max_noise
        sigma = random.uniform(0.0, config.max_noise)
        self.noise = self.generate_ou_noise(T=600, sigma=sigma)

        self.input_window = []          # To accumulate individual state vectors
        self.last_transition_index = None  # Holds index of last stored transition awaiting next_state

    def generate_ou_noise(self, T=600, mu=0, theta=0.15, sigma=0.01, dt=1):
        x = np.zeros(T)
        x[0] = 0
        for i in range(1, T):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        return x

    def average(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def calculate_epsilon(self):
        # Linearly decay epsilon over the designated epochs.
        # if self.current_epoch >= self.config.epsilon_decay_epochs:
        #     return self.config.final_epsilon
        # decay = (self.config.initial_epsilon - self.config.final_epsilon) / self.config.epsilon_decay_epochs
        # return self.config.initial_epsilon - decay * self.current_epoch
        return 1.0

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        v = max(0, v_ego_m_s)
        return math.sqrt(v) / math.sqrt(max_m_s)

    def map_action(self, action_index):
        # Map discrete action index (0 to action_space-1) to continuous steering torque.
        min_torque, max_torque = self.config.steering_torque_range
        return min_torque + (max_torque - min_torque) * (action_index / (self.config.action_space - 1))

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        # Construct state input as in the PID model.
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }
        previous_action = self.prev_actions[-3:] if len(self.prev_actions) >= 3 else [0, 0, 0]
        state_input_list = diff_values['lataccel'] + diff_values['roll'] + diff_values['a_ego'] + diff_values['v_ego'] + previous_action
        # Create a 1D tensor for the current state vector.
        current_state = torch.tensor(state_input_list, dtype=torch.float32)  # shape: (feature_dim,)

        # Append the current state to the input window.
        self.input_window.append(current_state)

        # If we haven't yet collected a full window, fall back to the PID model.
        if len(self.input_window) < self.config.window_size:
            action = self.pid_model.update(target_lataccel, current_lataccel, state, future_plan, steer)
            full_window = None
        else:
            # Build the full window from the most recent state vectors.
            full_window = torch.stack(self.input_window[-self.config.window_size:])  # shape: (window_size, feature_dim)
            # Add batch dimension: (1, window_size, feature_dim)
            full_window = full_window.unsqueeze(0)
            epsilon = self.calculate_epsilon()
            # Decide whether to use the Q-network or teacher.
            if self.q_model is not None and random.random() > epsilon:
                with torch.no_grad():
                    q_values = self.q_model(full_window)
                action_index = torch.argmax(q_values, dim=1).item()
                action = self.map_action(action_index)
            else:
                action = self.pid_model.update(target_lataccel, current_lataccel, state, future_plan, steer)

        # Add OU noise.
        noise_value = self.noise[len(self.prev_actions)] if len(self.noise) > len(self.prev_actions) else 0
        action += noise_value

        # If an explicit steer value is provided, override.
        if not math.isnan(steer):
            action = steer

        # Update the previous transition's next_state, if it exists.
        if self.last_transition_index is not None and full_window is not None:
            prev_trans = self.replay_buffer[self.last_transition_index]
            # Update the previous transition with the current full window as its next_state.
            self.replay_buffer[self.last_transition_index] = (prev_trans[0], prev_trans[1], prev_trans[2], full_window, prev_trans[4])
            self.last_transition_index = None

        # If we have a full window, store this transition.
        if full_window is not None:
            self.store_transition(full_window, action, reward=0.0, next_state=None, done=False)
            self.last_transition_index = len(self.replay_buffer) - 1
        else:
            # Optionally, you can choose to skip storing transitions until you have a full window.
            pass

        self.prev_actions.append(action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # Convert action to integer index if it's continuous
        if isinstance(action, float):  # Check if it's a float (steering torque)
            action = round((action - self.config.steering_torque_range[0]) /
                           (self.config.steering_torque_range[1] - self.config.steering_torque_range[0]) * (self.config.action_space - 1))

        # Ensure the action is within valid bounds
        action = max(0, min(action, self.config.action_space - 1))
        self.replay_buffer.append((state, action, reward, next_state, done))


def get_simulations(q_model, num_sims, config, current_epoch=0):
    # Initialize the physics model.
    model_path = Path('../../../models/tinyphysics.onnx')
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    # Get simulation files.
    file_list = [os.path.splitext(file)[0] for file in os.listdir("../../../data")]
    random.shuffle(file_list)
    file_list = file_list[:num_sims]

    results = defaultdict(dict)

    for file_name in tqdm(file_list):
        level_num = int(os.path.splitext(file_name)[0])
        data_path = os.path.join('../../../data/', f'{level_num:05}.csv')

        # Initialize the PID teacher.
        pid_controller = pid_model.Controller()
        # Create the dual controller.
        dual_controller = DualController(pid_controller, q_model, config, current_epoch)
        sim = tinyphysics.TinyPhysicsSimulator(
            tinyphysicsmodel, str(data_path),
            controller=dual_controller,
            debug=False
        )

        # Run the simulation.
        weights = [0.028, 0.141, 0.831]
        cost_history = []
        for t in range(20, len(sim.data)):
            sim.step()
            if t >= 101:
                total_cost = sim.compute_cost().get('total_cost')
                if not math.isnan(total_cost):
                    cost_history.append(total_cost)

                    # Ensure we have enough cost history for weighted diff calculation
                    if len(cost_history) >= 4:
                        # Calculate weighted score diff
                        weighted_cost = sum(weights[i] * (cost_history[-3 + i] - cost_history[-4 + i]) for i in range(3))

                        # Assign weighted diff to replay buffer (3 steps earlier)
                        last = dual_controller.replay_buffer[-1]
                        dual_controller.replay_buffer[-1] = (last[0], last[1], weighted_cost, last[3], last[4])


        results[file_name] = {"score": sim.compute_cost(), "buffer": dual_controller.replay_buffer}

    return results
