import os
import math
import random
from collections import defaultdict
from pathlib import Path

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.interactive(False)

import tinyphysics
from controllers import pid_model

SIM = None
model_path = Path('../../../models/tinyphysics.onnx')
random.seed(157)
np.random.seed(157)


class Controller:
    def __init__(self, internal_pid=None, max_noise=0.0):
        self.internal_pid = internal_pid
        self.prev_actions = []
        self.replay_buffer = []

        # Initialize windows for median calculations
        self.steer_window = []
        self.lataccel_window = []
        self.noise = self.generate_ou_noise(600, sigma=max_noise)

    def generate_ou_noise(self, T=0, mu=0, theta=0.15, sigma=0.01, dt=1):
        x = np.zeros(T)
        x[0] = 0
        for i in range(1, T):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        return x

    def store_transition(self, state, action):
        self.replay_buffer.append([state, action, 0.0, 0.0])

    def average(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        v_ego_m_s = max(0, v_ego_m_s)  # Sets negative values to 0
        return math.sqrt(v_ego_m_s) / math.sqrt(max_m_s)

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        global SIM

        # Compute the differences from the current state for each segment
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }

        # Previous steering torque
        previous_action = [0, 0, 0]
        if len(self.prev_actions) >= 3:
            previous_action = self.prev_actions[-3:]

        # Flatten the differences into a single list
        state_input_list = (diff_values['lataccel'] + diff_values['roll'] + diff_values['a_ego'] + diff_values['v_ego'] + previous_action)

        # Prepare the state as input for the model
        state_input = torch.tensor(state_input_list, dtype=torch.float32).unsqueeze(0)

        # Get action from controller
        action = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, steer)

        # Add noise to model action
        action += self.noise[len(self.prev_actions)]

        # Override initial steer commands
        if not math.isnan(steer):
            action = steer

        # Store transition in the replay buffer
        self.store_transition(state_input, action)

        self.prev_actions.append(action)
        return action

def get_simulations(max_noise=0.0, num_sims=0):
    global SIM
    tinyphysicsmodel = tinyphysics.TinyPhysicsModel(model_path, debug=False)

    # Get all simulations
    # file_list = json.load(open("levels.json", "r"))
    # if num_sims:
    #     # Shuffle and limit
    #     random.shuffle(file_list)
    #     file_list = file_list[:num_sims]
    file_list = [os.path.splitext(file)[0] for file in os.listdir("../../../data")]
    random.shuffle(file_list)
    file_list = file_list[:num_sims]

    results = defaultdict(dict)
    all_steer_costs = []
    existing_simulations = []

    for file_name in tqdm(file_list):
        level_num = int(os.path.splitext(file_name)[0])
        data_path = os.path.join('../../../data/', f'{level_num:05}.csv')

        if f'{level_num:05}' in existing_simulations:
            print(f"Skipping existing simulation: {level_num:05}")
            continue

        # Dictionary to store scores and replay buffers
        scores = {}
        torques = {}
        replay_buffers = {}

        # Run simulation
        controller_name = "PID-MODEL"
        internal_controller = pid_model.Controller()

        # Create simulator with the specific controller
        sim = tinyphysics.TinyPhysicsSimulator(
            tinyphysicsmodel, str(data_path),
            controller=Controller(internal_controller, max_noise),
            debug=False
        )

        # Run the simulation and calculate the cost
        previous_cost = 0.0
        cost_history = []  # Store cost history for weighted calculations
        weights = [0.028, 0.141, 0.831]  # Influence weights for the 3 time steps
        torque_list = []

        for _ in range(20, len(sim.data)):
            sim.step()
            if sim.controller.replay_buffer:
                torque_list.append(sim.controller.replay_buffer[-1][1])

            if _ >= 101:
                total_cost = sim.compute_cost().get('total_cost')
                if not math.isnan(total_cost):
                    # Append the current cost to the history
                    cost_history.append(total_cost)

                    # Ensure we have enough cost history for weighted diff calculation
                    if len(cost_history) >= 4:
                        # Calculate weighted score diff
                        weighted_diff = sum(weights[i] * (cost_history[-3 + i] - cost_history[-4 + i]) for i in range(3))
                        weighted_total = sum(weights[i] * cost_history[-3 + i] for i in range(3))

                        # Assign weighted diff to replay buffer (3 steps earlier)
                        if len(sim.controller.replay_buffer) >= 3:
                            sim.controller.replay_buffer[-3][2] = weighted_diff
                            sim.controller.replay_buffer[-3][3] = weighted_total
                            sim.controller.replay_buffer[-3][1] = [torque[1] for torque in sim.controller.replay_buffer[-3:]]

                    # Append the cost diff to all_steer_costs for analysis
                    cost_diff = total_cost - previous_cost
                    all_steer_costs.append(cost_diff)

                    # Update previous cost
                    previous_cost = total_cost

        # Compute the final cost for the controller
        cost = sim.compute_cost()
        torques[controller_name] = torque_list
        replay_buffer = sim.controller.replay_buffer

        # Save data for plotting
        results[file_name] = {"score": cost, "buffer": replay_buffer}

        # Output the score comparison
        #print(f"{level_num:05d}: {controller_name} : {cost['total_cost']:.1f} cost")

    return results


if __name__ == "__main__":
    results = get_simulations(max_noise=0.02)
