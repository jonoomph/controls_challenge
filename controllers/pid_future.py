from . import BaseController
import numpy as np
import math
from scipy.interpolate import CubicSpline


class Controller(BaseController):
    def __init__(self, p=0.15142, i=0.11571, d=-0.04128,
                 window_size=20, window_divisor=2.0, t_ahead=2.9502, road_roll_gain=0.37):
        """
        Initialize the PID controller with specified parameters.

        Parameters:
        p (float): Proportional gain.
        i (float): Integral gain.
        d (float): Derivative gain.
        window_size (int): Number of future points to consider.
        window_divisor (float): Divides window_size to determine the number of spline points.
        t_ahead (float): Time step ahead to predict the future lataccel.
        road_roll_gain (float): Gain applied to road roll compensation.
        """
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

    def weighted_future_lataccel(self, future_plan):
        """
        Predict the future lateral acceleration using a cubic spline based on the future plan.

        Parameters:
        future_plan: The plan containing future lataccel values.

        Returns:
        float: The interpolated future lateral acceleration.
        """
        actual_size = min(len(future_plan.lataccel), self.window_size)
        spline_points = round(actual_size / self.window_divisor)

        if actual_size <= 1 or spline_points < 2:
            return 0

        # Select points to better capture the curve
        indices = np.linspace(0, actual_size - 1, spline_points).astype(int)
        x_values = np.linspace(0, 1, len(indices))  # Normalize time steps for the spline
        y_values = [future_plan.lataccel[i] for i in indices]  # Get corresponding lataccel values

        # Fit a cubic spline to the selected points
        cubic_spline = CubicSpline(x_values, y_values)

        # Predict the lataccel at a small step ahead of the current value
        t = self.t_ahead * (1.0 / actual_size)  # Step towards the future
        interpolated_value = cubic_spline(t)

        return interpolated_value

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):
        """
        Update the controller state and compute the PID output.

        Parameters:
        target_lataccel (float): The target lateral acceleration.
        current_lataccel (float): The current lateral acceleration.
        state: The current state of the system, including roll_lataccel.
        future_plan: The plan containing future lataccel values.

        Returns:
        float: The PID controller output.
        """
        # Reset the error integral if the controller is not in use
        if current_lataccel == self.prev_target_lataccel:
            self.error_integral = 0.0

        # Predictive adjustment using future lataccel data
        predicted_lataccel = self.weighted_future_lataccel(future_plan)

        # Calculate the error and update integral and derivative terms
        error = predicted_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        self.prev_target_lataccel = target_lataccel

        # Road roll compensation based on the current state
        roll_compensation = state.roll_lataccel * self.road_roll_gain

        # Compute the PID output, subtracting roll compensation
        pid_output = (
                self.p * error +
                self.i * self.error_integral +
                self.d * error_diff -
                roll_compensation
        )
        return pid_output