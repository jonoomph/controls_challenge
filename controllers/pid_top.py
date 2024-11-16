# https://github.com/YashDYD/COMMA.AI-CONTROLS/blob/main/pid2_0.py
from . import BaseController
import numpy as np
import math


class Controller(BaseController):

    def __init__(self):
        self.p = 0.3
        self.i = 0.07
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0

        self.steer_factor = 13
        self.steer_sat_v = 20
        self.steer_command_sat = 2
        self.counter = 0

    def correct(self, action):
        current_output = self.p * (self.prev_error) + self.i * self.error_integral + self.d * (
                    self.prev_error - self.prev_error)
        correction = action - current_output  # Difference between external action and PID output

        # Adjust the integral and previous error to match the external action
        self.error_integral += correction / self.i if self.i != 0 else 0
        self.prev_error = correction / self.p if self.p != 0 else 0

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):
        self.counter += 1
        if self.counter == 81:
            self.error_integral = 0
            self.prev_error = 0

        # Optimized future lataccel average calculation with matching weights
        if len(future_plan.lataccel) >= 3:
            lataccel_combined = np.array([target_lataccel] + future_plan.lataccel[:3])
            weights = np.array([4, 5, 6, 7])  # Adjusted to match the length of lataccel_combined
            target_lataccel = np.average(lataccel_combined, weights=weights)

        # PID Error Calculation
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Dynamic scaling for high magnitude lataccel
        pid_factor = max(0.5, 1 - 0.23 * abs(target_lataccel))

        # Proportional gain dynamically adjusted for acceleration
        p_dynamic = max(0.1, self.p - 0.1 * abs(state.a_ego))

        # PID Control signal
        u_pid = (p_dynamic * error + self.i * self.error_integral + self.d * error_diff) * pid_factor

        # Feedforward control: Adjusted with a sigmoid function for smoothness
        steer_accel_target = target_lataccel - state.roll_lataccel
        steer_command = steer_accel_target * self.steer_factor / max(self.steer_sat_v, state.v_ego)
        steer_command = 2 * self.steer_command_sat / (1 + math.exp(-steer_command)) - self.steer_command_sat

        # Combined control signal with feedforward gain
        u_ff = 0.8 * steer_command

        return np.clip(u_pid + u_ff, -2, 2)
