from . import BaseController
import pygame

# Initialize PyGame and the PS5 controller
pygame.init()
pygame.joystick.init()

# Set up the clock to control the FPS
clock = pygame.time.Clock()
FPS = 10  # Match simulation FPS

# Assuming there is only one joystick (your PS5 controller)
joystick = pygame.joystick.Joystick(0)
joystick.init()

class Controller(BaseController):
    def __init__(self):
        self.step_idx = -1

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        pygame.event.pump()

        # Get raw joystick input [-1, 1]
        raw_input = -joystick.get_axis(0)

        # Apply non-linear mapping
        # Choose an exponent greater than 1 for less sensitivity around the center
        exponent = 3  # Adjust this value to change the curve
        torque_output = self.non_linear_mapping(raw_input, exponent)

        self.step_idx += 1

        return torque_output

    def non_linear_mapping(self, x, exponent):
        # Apply the exponent while preserving the sign
        return (abs(x) ** exponent) * (1 if x >= 0 else -1)
