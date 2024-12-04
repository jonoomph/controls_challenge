import onnxruntime as ort
import numpy as np
from . import BaseController
import math


class Controller(BaseController):
    """
    AI-powered PID controller with error correction via traditional PID logic.
    """

    def correct(self, action):
        pass

    def __init__(self, window_size=30, model_path="/home/jonathan/apps/controls_challenge/game/train/onnx/model-wONff-24.onnx"):
        """
        Initialize the controller with a specified ONNX model and time-series window size.

        Args:
            window_size (int): Size of the time-series window for model input.
            model_path (str): Path to the ONNX model file.
        """
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = [
            #'CUDAExecutionProvider',
            'CPUExecutionProvider']

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, providers)

        # Initialize parameters
        self.window_size = window_size
        self.input_window = []
        self.prev_actions = []
        self.step_idx = 20

        # Initialize windows for median calculations
        self.steer_window = []
        self.lataccel_window = []

    def average(self, values):
        """ Calculate the average of a list of values. """
        return sum(values) / len(values) if values else 0

    def normalize_v_ego(self, v_ego_m_s):
        max_m_s = 40.0
        v_ego_m_s = max(0, v_ego_m_s)  # Sets negative values to 0
        return math.sqrt(v_ego_m_s) / math.sqrt(max_m_s)

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        """
        Update the control signal based on the current state and future plan.

        Args:
            target_lataccel (float): Target lateral acceleration.
            current_lataccel (float): Current lateral acceleration.
            state (object): Current state object containing roll, speed, and acceleration.
            future_plan (object): Predicted future states for the vehicle.

        Returns:
            float: Control signal for steering.
        """
        # Calculate differences for future segments
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

        state_input = np.array(diff_values['lataccel'] + diff_values['roll'] + diff_values['a_ego'] + diff_values['v_ego'] + previous_action, dtype=np.float32)

        # Update time-series window
        self.input_window.append(state_input)

        # Run model if window is ready
        control_signal = 0
        if len(self.input_window) >= self.window_size: # and self.step_idx >= 93:
            input_tensor = np.array(self.input_window[-self.window_size:]).reshape(1, self.window_size, -1)
            output = self.ort_session.run(None, {'input': input_tensor})[0]
            control_signal = output[0,0]

            # distance_from_target = abs(current_lataccel - self.average(future_plan.lataccel[1:3]))
            # lower_bound = 0.9
            # upper_bound = 0.99
            #
            # # Calculate the weight for output[0,1]
            # weight = np.clip((distance_from_target - lower_bound) / (upper_bound - lower_bound), 0, 1)
            #
            # # Interpolate between output[0,0] and output[0,1]
            # control_signal = (1 - weight) * output[0, 0] + weight * output[0, 1]

        # Override initial steer values
        if not math.isnan(steer):
            control_signal = steer

        # Save action and update step index
        self.prev_actions.append(control_signal)
        self.step_idx += 1

        return control_signal
