import math
import torch

from . import BaseController
from . import experimental, pid_model
from game.train.deep.qmodel import QSteeringNet


class Controller(BaseController):
    """
    AI-powered PID controller using a PyTorch Q-model for prediction.
    Initially, the controller uses a traditional PID (via experimental.Controller)
    for the early steps. Once a full window of input states is available and after
    a specific simulation step index, the controller uses the Q-network's argmax
    prediction. The chosen action index is then mapped to a continuous steering
    torque in the range [-2.5, 2.5].
    """

    def __init__(self, model_path="/home/jonathan/apps/controls_challenge/game/train/deep/onnx/model-UaGuX-22.pth",
                 window_size=30, action_space=257):
        """
        Args:
            model_path (str): Path to the saved PyTorch model (.pth file).
            window_size (int): Number of recent states to form the input window.
            action_space (int): Number of discrete actions (default 257 maps from -2.5 to 2.5).
        """
        # 1) Build your Q-network and load weights.
        self.model = QSteeringNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()  # Set to eval mode (no dropout, etc.)

        # Controller parameters
        self.window_size = window_size
        self.input_window = []  # Sliding window for input state vectors
        self.prev_actions = []  # History of previous control signals
        self.step_idx = 20  # Simulation step index
        self.internal_pid = experimental.Controller()
        self.internal_teacher_pid = pid_model.Controller()
        self.takeover = False
        self.action_space = action_space
        self.min_torque = -2.5
        self.max_torque = 2.5

    def average(self, values):
        """Calculate the average of a list of values, handling empty lists gracefully."""
        return sum(values) / len(values) if values else 0

    def normalize_v_ego(self, v_ego_m_s):
        """
        Normalize vehicle speed to a range [0, 1] using square root scaling.
        """
        max_m_s = 40.0
        v = max(0, v_ego_m_s)
        return math.sqrt(v) / math.sqrt(max_m_s)

    def map_action(self, action_index):
        """
        Map a discrete action index to a continuous steering torque.
        For 257 actions, index 0 => -2.5, index 256 => +2.5.
        """
        return self.min_torque + (self.max_torque - self.min_torque) * (action_index / (self.action_space - 1))

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.nan):
        """
        Update the control signal using the Q-model when possible, otherwise fall back to PID.

        Args:
            target_lataccel (float): Target lateral acceleration.
            current_lataccel (float): Current lateral acceleration.
            state (object): Current vehicle state containing roll, speed, etc.
            future_plan (object): Predicted future states.
            steer (float): Optional manual steer override (default NaN).

        Returns:
            float: Control signal (steering torque).
        """
        # 1) Get PID signals
        pid_teacher = self.internal_teacher_pid.update(target_lataccel, current_lataccel, state, future_plan, math.nan)
        pid_steer = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, math.nan)

        # 2) Build state input vector from differences
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[s:e]) for s, e in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[s:e]) for s, e in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[s:e])) for s, e in future_segments],
            'a_ego': [self.average(future_plan.a_ego[s:e]) for s, e in future_segments],
        }
        prev_action = self.prev_actions[-3:] if len(self.prev_actions) >= 3 else [0, 0, 0]
        state_input_list = (diff_values['lataccel'] + diff_values['roll'] + diff_values['a_ego'] + diff_values['v_ego'] + prev_action)

        # 3) Append current state vector to the sliding window
        self.input_window.append(state_input_list)

        # 4) By default, set control signal to 0
        control_signal = 0.0

        # 5) If we have a full window, run inference with our PyTorch model
        if len(self.input_window) >= self.window_size:
            # Convert the last window_size rows to a torch tensor: (1, window_size, feature_dim)
            window_data = self.input_window[-self.window_size:]
            # shape: (window_size, feature_dim)
            input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
            # shape: (1, window_size, feature_dim)

            with torch.no_grad():
                q_values = self.model(input_tensor)  # shape: (1, action_space)
                action_index = torch.argmax(q_values, dim=1).item()

            # Map discrete action index to continuous torque
            control_signal = self.map_action(action_index)

            # DEBUG
            #control_signal = pid_teacher
            print(control_signal, pid_teacher)

        # 6) Until a certain simulation step, rely on the PID
        if self.step_idx < 105:
            control_signal = pid_steer

        # 7) Save control signal and increment step index
        self.prev_actions.append(control_signal)
        self.step_idx += 1
        return control_signal
