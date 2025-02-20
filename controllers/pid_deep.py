import onnxruntime as ort
import numpy as np
from . import BaseController
from . import experimental, pid_model
import math


class Controller(BaseController):
    """
    AI-powered PID controller using an ONNX Q-model for prediction.
    Initially, the controller uses a traditional PID (via experimental.Controller)
    for the early steps. Once a full window of input states is available and after a specific
    simulation step index, the controller uses the Q-network's argmax prediction.
    The chosen action index is then mapped to a continuous steering torque in the range [-2.5, 2.5].
    """

    def __init__(self, window_size=30, model_path="/home/jonathan/apps/controls_challenge/game/train/deep/onnx/model-UaGuX-12.onnx", action_space=257):
        """
        Args:
            window_size (int): Number of recent states to form the input window.
            model_path (str): Path to the exported ONNX Q-model.
            action_space (int): Number of discrete actions (default 257 maps from -2.5 to 2.5).
        """
        # Set up ONNX runtime session for the Q-model.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = ['CPUExecutionProvider']
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, providers)

        # Controller parameters.
        self.window_size = window_size
        self.input_window = []  # Sliding window for input state vectors.
        self.prev_actions = []  # History of previous control signals.
        self.step_idx = 20  # Simulation step index.
        self.internal_pid = experimental.Controller()
        self.internal_teacher_pid = pid_model.Controller()
        self.takeover = False
        self.action_space = action_space  # Discrete action space size.
        self.min_torque = -2.5  # Minimum steering torque.
        self.max_torque = 2.5  # Maximum steering torque.

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

        Args:
            action_index (int): Discrete action index (0 to action_space-1).

        Returns:
            float: Steering torque in the range [-2.5, 2.5].
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
        # Compute the PID control signal using the experimental PID controller.
        pid_teacher = self.internal_teacher_pid.update(target_lataccel, current_lataccel, state, future_plan, math.nan)
        pid_steer = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, math.nan)

        # Build state input vector from differences.
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }
        previous_action = self.prev_actions[-3:] if len(self.prev_actions) >= 3 else [0, 0, 0]
        state_input = np.array(
            diff_values['lataccel'] +
            diff_values['roll'] +
            diff_values['a_ego'] +
            diff_values['v_ego'] +
            previous_action,
            dtype=np.float32
        )
        # Append current state vector to the sliding window.
        self.input_window.append(state_input)

        # By default, set control signal to 0.
        control_signal = 0.0

        # Once we have a full window, use the Q-model for prediction.
        if len(self.input_window) >= self.window_size:
            # Construct the input tensor: shape (1, window_size, feature_dim)
            input_tensor = np.array(self.input_window[-self.window_size:]).reshape(1, self.window_size, -1)
            # Run the ONNX model to get Q-values.
            output = self.ort_session.run(None, {'input': input_tensor})[0]
            # Get the index of the maximum Q-value.
            action_index = int(np.argmax(output, axis=1)[0])
            # Map the discrete action index to continuous torque.
            control_signal = self.map_action(action_index)

            # DEBUG
            #control_signal = pid_teacher
            print(self.map_action(action_index), pid_teacher)

        # Until a certain simulation step, use the PID controller.
        if self.step_idx < 105:
            control_signal = pid_steer

        # Save control signal and increment step index.
        self.prev_actions.append(control_signal)
        self.step_idx += 1
        return control_signal
