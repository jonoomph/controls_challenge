import onnxruntime as ort
import numpy as np
import pandas as pd
from scipy.stats import linregress
from . import BaseController
import math


class Controller(BaseController):
    """
    AI-powered PID controller with error correction via traditional PID logic.
    """

    def __init__(self, window_size=22, model_path="/home/jonathan/apps/controls_challenge/game/train/onnx/lat_accel_predictor-SumWJ-12.onnx"):
        """
        Initialize the controller with a specified ONNX model and time-series window size.

        Args:
            window_size (int): Size of the time-series window for model input.
            model_path (str): Path to the ONNX model file.
        """
        # Load ONNX model
        self.ort_session = ort.InferenceSession(model_path)

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
        """ Normalize the vehicle's speed for model input. """
        return v_ego_m_s / 40.0

    def calculate_median_r2(self, torque, lataccel, window_size=22):
        """ Calculate the median R² and slope from the sliding window of data. """
        self.steer_window.append(torque)
        self.lataccel_window.append(lataccel)

        # Keep windows within size limits
        if len(self.steer_window) > window_size:
            self.steer_window.pop(0)
        if len(self.lataccel_window) > window_size:
            self.lataccel_window.pop(0)

        # Calculate and print median R² if window is ready
        if len(self.steer_window) >= window_size:
            # Perform linear regression on the sliding window data
            steer_command = pd.Series(self.steer_window)
            lat_accel = pd.Series(self.lataccel_window)

            # Calculate the slope and R² (correlation and regression)
            try:
                slope, intercept, r_value, p_value, std_err = linregress(steer_command, lat_accel)
                return r_value**2, slope
            except ValueError:
                pass
        return 0, 0

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
        future_segments = [(0, 1), (1, 3), (2, 5), (5, 9), (9, 14), (14, 20)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(state.v_ego) - self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [state.a_ego - self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }

        # Prepare state input for the model
        previous_action = self.prev_actions[-1] if self.prev_actions else 0

        # Append values to sliding windows for steer and target lateral acceleration
        r2, slope = self.calculate_median_r2(previous_action, current_lataccel, self.window_size)

        state_input = np.array(
            diff_values['lataccel'] + diff_values['roll'] + diff_values['v_ego'] + diff_values['a_ego'] +
            [previous_action], dtype=np.float32)

        # Update time-series window
        self.input_window.append(state_input)

        # Run model if window is ready
        control_signal = 0
        if len(self.input_window) >= self.window_size: # and self.step_idx >= 93:
            input_tensor = np.array(self.input_window[-self.window_size:]).reshape(1, self.window_size, -1)
            control_signal = self.ort_session.run(None, {'input': input_tensor})[0][0, 0]

        # Override initial steer values
        if not math.isnan(steer):
            control_signal = steer

        # Save action and update step index
        self.prev_actions.append(control_signal)
        self.step_idx += 1

        return control_signal
