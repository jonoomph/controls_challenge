from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan, steer=None):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff

  def correct(self, action):
      current_output = self.p * (self.prev_error) + self.i * self.error_integral + self.d * (self.prev_error - self.prev_error)
      correction = action - current_output  # Difference between external action and PID output

      # Adjust the integral and previous error to match the external action
      self.error_integral += correction / self.i if self.i != 0 else 0
      self.prev_error = correction / self.p if self.p != 0 else 0
