# https://github.com/avangerp/controls_challenge/tree/master/controllers
from . import BaseController
import numpy as np
import math


class Controller(BaseController):

    def __init__(self, ):
        self.p = 0.3
        self.i = 0.07
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0

        self.steer_factor = 13  # lat accel to steer command factor
        self.steer_sat_v = 20  # saturate v measurements for steering
        self.steer_command_sat = 2  # feedforward command magnitude saturation
        self.counter = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):

        # reset error integral and prev error variables when the controller is first given control authority
        self.counter += 1
        if self.counter == 81:
            self.error_integral = 0
            self.prev_error = 0

        # get weighted avg of current + future target lataccel because input response is slow
        if len(future_plan.lataccel) >= 3:
            target_lataccel = np.average([target_lataccel] + future_plan.lataccel[0:3], weights=[5, 6, 7, 8])

        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # scale pid input down for high magnitude target lat accel
        pid_factor = min(1, 1 - (abs(target_lataccel) - 1) * 0.23)

        # scale p gain down for high magnitude long accel
        p = (self.p - abs(state.a_ego) / 10)

        # pid input
        u_pid = (p * error + self.i * self.error_integral + self.d * error_diff) * pid_factor

        # estimate some steer command based on available measurements
        steer_accel_target = (target_lataccel - state.roll_lataccel)
        steer_command = (steer_accel_target * self.steer_factor /
                         max(self.steer_sat_v, state.v_ego))

        # sigmoid feed forward to reduce jerk
        steer_command = 2 * self.steer_command_sat / (1 + math.exp(-steer_command)) - self.steer_command_sat

        K_ff = 0.8  # feed forward input gain
        u_ff = K_ff * steer_command

        return u_pid + u_ff