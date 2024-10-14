import pygame
import tinyphysics
from tinyphysics import CONTEXT_LENGTH
from controllers import BaseController

# Initialize PyGame and the PS5 controller
pygame.init()
pygame.joystick.init()

# Assuming there is only one joystick (your PS5 controller)
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Set up the clock to control the FPS
clock = pygame.time.Clock()
FPS = 30  # Set the desired frames per second


class Controller(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Capture the current PS5 controller's left stick input
        pygame.event.pump()  # Necessary to process events and get current state of the controller

        # Axis 0 is left stick X-axis, we assume this for torque [-1, 1]
        left_stick_x = joystick.get_axis(0)  # Range is usually [-1, 1] for analog sticks

        # Map the stick's X position to torque (which is already [-1, 1] here)
        torque = left_stick_x

        # Print out the current torque from the controller
        print(f"Torque: {torque}")

        # For now, just return the torque value from the controller
        return torque


DEBUG = True
DATA_PATH = "../data/00000.csv"
MODEL_PATH = "../models/tinyphysics.onnx"
CONTROLLER = Controller()
MODEL = tinyphysics.TinyPhysicsModel(MODEL_PATH, debug=DEBUG)
SIM = tinyphysics.TinyPhysicsSimulator(MODEL, str(DATA_PATH), controller=CONTROLLER, debug=DEBUG)

# Initialize step index to start from CONTEXT_LENGTH
SIM.step_idx = CONTEXT_LENGTH

# Game Loop with FPS limiting
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exit the game loop if the window is closed

    # Simulation step only if we're within the valid data range
    if SIM.step_idx < len(SIM.data):
        SIM.step()

        # Debug print every 10 steps
        if SIM.debug and SIM.step_idx % 10 == 0:
            print(
                f"Step {SIM.step_idx:<5}: Current lataccel: {SIM.current_lataccel:>6.2f}, Target lataccel: {SIM.target_lataccel_history[-1]:>6.2f}")

    # Stop running when simulation reaches the end of the data
    else:
        running = False

    # Limit the frame rate to FPS
    clock.tick(FPS)

# Close out PyGame when done
pygame.quit()
