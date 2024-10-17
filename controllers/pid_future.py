from . import BaseController
import numpy as np
from scipy.interpolate import CubicSpline


class Controller(BaseController):
    def __init__(self, p=0.15142857142857144, i=0.11571428571428571, d=-0.04128571428571429, window_size=20, window_divisor=2.0, t_ahead=2.9502, road_roll_gain=0.37, file_index=0, increment=0.08868686868686868):
        self.p = p
        self.i = i
        self.d = d
        self.window_size = window_size
        self.window_divisor = window_divisor
        self.t_ahead = t_ahead
        self.road_roll_gain = road_roll_gain
        self.error_integral = 0
        self.prev_error = 0
        self.prev_target_lataccel = 0
        self.prev_action = 0
        self.rounded_pid_diff = 0
        self.step_idx = 20
        self.file_index = file_index
        self.initial_steer = []
        self.increment = increment

    def weighted_future_lataccel(self, future_plan):
        # Use all available future values up to window_size
        actual_size = min(len(future_plan.lataccel), self.window_size)
        spline_points = round(actual_size / self.window_divisor)
        if actual_size <= 1 or spline_points < 2:
            return 0

        # Choose more points to better capture the curve
        indices = np.linspace(0, actual_size - 1, spline_points).astype(int)
        x_values = np.linspace(0, 1, len(indices))  # Normalize the time steps for these points
        y_values = [future_plan.lataccel[i] for i in indices]  # Select the corresponding lataccel values

        # Fit a cubic spline to the selected points using 'natural' boundary condition
        cubic_spline = CubicSpline(x_values, y_values)

        # Evaluate the spline at a small step ahead of the current value
        t = self.t_ahead * (1.0 / actual_size)  # Small step towards the future

        interpolated_value = cubic_spline(t)
        return interpolated_value

    def adjust_based_on_model_action(self, new_action_diff):
        self.prev_action -= self.rounded_pid_diff
        # self.rounded_pid_diff = new_action_diff  # Remove or comment out this line
        self.prev_action += new_action_diff

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Predictive adjustment using future plan data
        predicted_lataccel = self.weighted_future_lataccel(future_plan)

        # Calculate error based on the predicted lateral acceleration
        error = predicted_lataccel - current_lataccel

        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Road roll compensation
        roll_compensation = state.roll_lataccel * self.road_roll_gain

        # Reset error integral when controller not in use
        if self.step_idx < 80:
            error = 0
            self.error_integral = 0
            self.prev_error = 0
            error_diff = 0

        # Compute the PID output using the adaptive gains and roll compensation
        pid_output = (
                self.p * error +
                self.i * self.error_integral +
                self.d * error_diff -
                roll_compensation  # Subtract roll compensation to counteract its effect
        )

        # Increment step
        self.step_idx += 1

        # Calculate the difference between the current PID output and the previous action
        pid_diff = pid_output - self.prev_action

        # Round the PID difference to the nearest allowed increment
        self.rounded_pid_diff = min(max(round(pid_diff / self.increment) * self.increment, -self.increment), self.increment)

        # Update the rounded PID output by adding the rounded difference to the previous action
        rounded_pid_output = self.prev_action + self.rounded_pid_diff

        # Update the previous action to the current rounded output
        self.prev_action = rounded_pid_output
        return rounded_pid_output
