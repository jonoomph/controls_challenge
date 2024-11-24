from . import BaseController
import numpy as np
import math


class Controller(BaseController):

    def __init__(self):
        # PID gains
        self.p = 0.3
        self.i = 0.07
        self.d = -0.1

        # Error terms
        self.error_integral = 0
        self.prev_error = 0

        # Steering and feedforward parameters
        self.steer_factor = 13
        self.steer_sat_v = 20
        self.steer_command_sat = 2
        self.counter = 0
        self.max_integral = 10  # Anti-windup limit for integral term

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):
        # Reset error integral and prev error when control is handed over
        self.counter += 1
        if self.counter == 81:
            self.error_integral = 0
            self.prev_error = 0

        # Dynamically weighted average for future lataccel
        if len(future_plan.lataccel) >= 3:
            velocity_factor = min(1, state.v_ego / 30)  # Scale between 0 and 1
            weights = [5 + velocity_factor, 6 + velocity_factor, 7 + velocity_factor, 8 + velocity_factor]
            target_lataccel = np.average([target_lataccel] + future_plan.lataccel[0:3], weights=weights)

        # PID error calculation
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Anti-windup for integral term
        if self.error_integral > self.max_integral:
            self.error_integral = self.max_integral
        elif self.error_integral < -self.max_integral:
            self.error_integral = -self.max_integral

        # Dynamic scaling of PID gains
        pid_factor = max(0.5, 1 - 0.23 * abs(target_lataccel))
        p_dynamic = max(0.1, self.p - 0.1 * abs(state.a_ego))
        i_dynamic = self.i * pid_factor
        d_dynamic = self.d * (1 + 0.1 * abs(state.a_ego))  # Slightly increase D for higher accelerations

        # PID control signal
        u_pid = (p_dynamic * error + i_dynamic * self.error_integral + d_dynamic * error_diff) * pid_factor

        # Feedforward control: adaptive gain based on speed and curvature
        steer_accel_target = target_lataccel - state.roll_lataccel
        steer_command = steer_accel_target * self.steer_factor / max(self.steer_sat_v, state.v_ego)
        steer_command = 2 * self.steer_command_sat / (1 + math.exp(-steer_command)) - self.steer_command_sat

        # Adaptive feedforward gain
        K_ff = 0.8 + 0.1 * (abs(state.v_ego) / 30)  # Increase feedforward gain slightly with speed
        u_ff = K_ff * steer_command

        # Combined control output
        return np.clip(u_pid + u_ff, -2, 2)
