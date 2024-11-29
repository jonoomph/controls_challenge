from . import BaseController
import math

class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def update(self, target_lataccel, current_lataccel, state, future_plan, steer=math.inf):
    return 0.0
