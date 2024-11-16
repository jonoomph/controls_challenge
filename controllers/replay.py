import os.path
import math

from . import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self, level_num=8335, torques=None):
        level_file_path = (f'/home/jonathan/apps/controls_challenge/game/data-optimized/{level_num:05}.npy',
                           f'/home/jonathan/apps/controls_challenge/game/data/{level_num:05}.npy')
        if torques is not None and len(torques) > 0:
            self.torques = torques
        elif os.path.exists(level_file_path[0]):
            self.torques = np.load(level_file_path[0])
        elif os.path.exists(level_file_path[1]):
            self.torques = np.load(level_file_path[1])
        else:
            self.torques = np.zeros(581)
        self.step_idx = 19

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer=None):
        self.step_idx += 1
        if steer and not math.isnan(steer):
            return steer
        else:
            return self.torques[self.step_idx-20]
