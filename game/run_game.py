import pygame
import tinyphysics
import numpy as np
from tinyphysics import CONTEXT_LENGTH
from controllers import BaseController, pid
import os
import json


# Initialize PyGame and the PS5 controller
pygame.init()
pygame.joystick.init()

# Set up the screen dimensions and colors
WIDTH, HEIGHT = 600, 1200  # Tall window size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
TEAL = (0, 40, 100)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Simulation")

# Set up the clock to control the FPS
clock = pygame.time.Clock()
BASE_FPS = 8
SCALE_FPS = False

# Assuming there is only one joystick (your PS5 controller)
# joystick = pygame.joystick.Joystick(0)
# joystick.init()

# Load images
background_image = pygame.image.load("images/background.png").convert()
car_image = pygame.image.load("images/car.png").convert_alpha()
road_image = pygame.image.load("images/road.png").convert_alpha()
road_checker_image = pygame.image.load("images/road-checker.png").convert_alpha()
wheel_image = pygame.image.load("images/wheel.png").convert_alpha()


# Car constants
CAR_WIDTH, CAR_HEIGHT = 15, 25  # Car size adjusted to match car image
ROAD_WIDTH = 600  # Road width matches the road image width
SEGMENT_HEIGHT = 25  # Matches the height of the road image
MAX_LATACCEL_DIFF = 5  # Maximum expected absolute value of lateral acceleration difference
ROAD_AGGRESSIVE_FACTOR = 160  # Adjust this value as needed for road shift sensitivity

# Function to draw the car
def draw_car(screen, car_x, car_y, rotation_angle):
    rotated_car = pygame.transform.rotate(car_image, rotation_angle)
    screen.blit(rotated_car, rotated_car.get_rect(center=(car_x, car_y)))

def draw_road(screen, future_plan, car_y, current_lataccel, target_lataccel, roll_lataccel, index):
    lataccel_diff = current_lataccel - target_lataccel
    road_center_x = (WIDTH // 2) + (lataccel_diff * ROAD_AGGRESSIVE_FACTOR)

    # Draw the road directly under the car
    road_x = road_center_x - (ROAD_WIDTH // 2)
    current_segment_y = car_y
    screen.blit(road_image, (road_x, current_segment_y))

    # Draw the arrow for current road roll
    draw_arrow(screen, road_x, current_segment_y, roll_lataccel)

    # Draw future segments with road roll arrows
    for i, fut_lataccel in enumerate(future_plan.lataccel):
        lataccel_diff = current_lataccel - fut_lataccel
        road_center_x = (WIDTH // 2) + (lataccel_diff * ROAD_AGGRESSIVE_FACTOR)
        segment_y = car_y - (i + 1) * SEGMENT_HEIGHT  # Future road segments

        road_x = road_center_x - (ROAD_WIDTH // 2)
        if index + i in [80, 577]:
            screen.blit(road_checker_image, (road_x, segment_y))
        else:
            screen.blit(road_image, (road_x, segment_y))

        # Draw the arrow for future road rolls
        future_road_roll = future_plan.roll_lataccel[i]  # Assuming road rolls match lataccel
        draw_arrow(screen, road_x, segment_y, future_road_roll)

def draw_steering(screen, torque_value):
    # Map torque value (-2 to 2) to degrees (-360 to 360)
    rotation_angle = -torque_value * 90  # Each torque unit corresponds to 180 degrees (since 2*180=360)

    # Rotate the wheel around its center
    rotated_wheel = pygame.transform.rotate(wheel_image, -rotation_angle)  # Negative to match correct direction
    wheel_rect = rotated_wheel.get_rect(center=(100, HEIGHT - 100))  # Position in bottom-left corner

    # Draw the rotated wheel on the screen
    screen.blit(rotated_wheel, wheel_rect)

    # Render the torque value as text
    font = pygame.font.Font(None, 32)
    torque_text = font.render(f"{torque_value:.2f}", True, WHITE)
    text_rect = torque_text.get_rect(center=wheel_rect.center)

    # Draw the torque text in the middle of the wheel (unrotated)
    screen.blit(torque_text, text_rect)

def draw_arrow(screen, road_x, road_y, road_roll):
    # Constants
    max_arrow_length = 200  # Max arrow length in pixels
    arrow_height = 2        # Height of the arrow line in pixels
    triangle_size = 10      # Size of the triangle at the end of the arrow
    threshold = 0.0        # Threshold below which we don't show the triangle

    # Calculate the length of the arrow based on road_roll magnitude
    arrow_length = max_arrow_length * abs(road_roll)

    # Define start position of the arrow (right side of the road)
    start_x = road_x + ROAD_WIDTH // 2 + 120  # 20px to the right of the road
    start_y = road_y

    # Define end position based on the sign of the road_roll
    if road_roll > 0:
        end_x = start_x + arrow_length  # Arrow to the right
    else:
        end_x = start_x - arrow_length  # Arrow to the left

    # Draw the triangle if abs(road_roll) >= threshold
    if abs(road_roll) >= threshold:
        # Draw the horizontal line (2px tall)
        pygame.draw.line(screen, WHITE, (start_x, start_y), (end_x, start_y), arrow_height)

        # Triangle points to the direction of the roll
        if road_roll > 0:  # Roll to the right
            triangle_points = [(end_x, start_y),
                               (end_x - triangle_size, start_y - triangle_size // 2),
                               (end_x - triangle_size, start_y + triangle_size // 2)]
        else:  # Roll to the left
            triangle_points = [(end_x, start_y),
                               (end_x + triangle_size, start_y - triangle_size // 2),
                               (end_x + triangle_size, start_y + triangle_size // 2)]

        # Draw the triangle
        pygame.draw.polygon(screen, WHITE, triangle_points)

# Generate sigmoid-based torque levels
def sigmoid_space(min_value, max_value, num_steps):
    x = np.linspace(-6, 6, num_steps)  # Create a range of values centered around 0
    sigmoid = 1 / (1 + np.exp(-x))  # Sigmoid function
    return min_value + (max_value - min_value) * sigmoid


class Controller(BaseController):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.internal_pid = pid.Controller()
        self.torques = []
        self.lat_accel_cost = 0
        self.jerk_cost = 0
        self.total_cost = 0

        # Initialize torque levels for 1024 steps
        self.min_torque = -2.0
        self.max_torque = 2.0
        self.num_steps = 80
        self.torque_levels = np.linspace(self.min_torque, self.max_torque, self.num_steps)
        self.current_torque_index = self.num_steps // 2  # Start at zero torque

        # Initialize font for displaying score
        self.font = pygame.font.Font(None, 36)
        self.level_font = pygame.font.Font(None, 50)

    def update_score(self, score):
        # Update the stored score values
        self.lat_accel_cost = score['lataccel_cost']
        self.jerk_cost = score['jerk_cost']
        self.total_cost = score['total_cost']

    def draw_score(self):
        # Create the text surfaces
        lat_text = self.font.render(f"Lat: {self.lat_accel_cost:.2f}", True, WHITE)
        jerk_text = self.font.render(f"Jerk: {self.jerk_cost:.2f}", True, WHITE)
        total_text = self.font.render(f"Total: {self.total_cost:.2f}", True, WHITE)

        # Position the text at the top of the screen
        screen.blit(lat_text, (20, 20))
        screen.blit(jerk_text, (20, 60))
        screen.blit(total_text, (20, 100))

    def draw_level(self, level_num):
        # Create the text surface for the level number
        level_text = self.level_font.render(f"Level: {level_num}", True, WHITE)

        # Position the level text at the top-right of the screen
        screen.blit(level_text, (WIDTH - level_text.get_width() - 20, 20))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        pygame.event.pump()  # Necessary to process events
        pid_action = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan)

        # Get raw joystick input [-1, 1]
        #raw_input = -joystick.get_axis(0)
        # Apply non-linear mapping
        #exponent = 3  # Adjust this value to change the curve
        #torque_output = pid_action + self.non_linear_mapping(raw_input, exponent) * 1.5

        # Determine position on road and SIM index
        lataccel_diff = current_lataccel - target_lataccel
        index = len(self.torques)
        increment = 1

        # # Get the state of all keyboard buttons
        keys = pygame.key.get_pressed()

        # Calculate FPS based on vego
        if SCALE_FPS:
            FPS = 10.0 * (state.v_ego / 22.0)
        else:
            FPS = BASE_FPS

        # Increment
        if keys[pygame.K_LCTRL]:
            FPS /= 3
        if keys[pygame.K_LSHIFT]:
            FPS *= 2
        clock.tick(FPS)

        # Adjust torque index based on key presses
        if keys[pygame.K_LEFT]:
            # Decrease torque index (turn left)
            self.current_torque_index = min(max(self.current_torque_index + increment, 0), self.num_steps - 1)
        elif keys[pygame.K_RIGHT]:
            # Increase torque index (turn right)
            self.current_torque_index = min(max(self.current_torque_index - increment, 0), self.num_steps - 1)
        else:
            # Gradually move towards target (slow auto-steer)
            if index % 5 == 0 and abs(lataccel_diff) > 0.10:
                if lataccel_diff > 0.0:
                    self.current_torque_index -= 1
                else:
                    self.current_torque_index += 1

        # Get the torque output from the torque levels array
        torque_output = self.torque_levels[self.current_torque_index]# + pid_action
        #print(lataccel_diff, torque_output)

        # Rotate the car based on current lateral acceleration
        car_rotation = torque_output * 10  # Adjust the multiplier for realistic rotation

        # Draw everything
        screen.blit(background_image, (0, 0))

        # Draw the road based on future lateral acceleration and target alignment
        draw_road(screen, future_plan, HEIGHT - 100, current_lataccel, target_lataccel, state.roll_lataccel, index)
        draw_car(screen, WIDTH // 2, HEIGHT - 100, car_rotation)
        draw_steering(screen, torque_output)

        # Draw the score at the top
        self.draw_score()
        self.draw_level(self.level)

        # Update display
        pygame.display.flip()

        # Return torque value to simulator (not used in drawing)
        self.torques.append(torque_output)
        return torque_output


DEBUG = True
LEVEL_NUM = 124
TINY_DATA_DIR = "../data"
GAME_DATA_DIR = "data"
SCORES_FILE = os.path.join(GAME_DATA_DIR, "high_scores.json")
DATA_PATH = os.path.join(TINY_DATA_DIR, f"{LEVEL_NUM:05}.csv")
MODEL_PATH = "../models/tinyphysics.onnx"

# Load high scores or initialize if it doesn't exist
if os.path.exists(SCORES_FILE):
    with open(SCORES_FILE, "r") as f:
        high_scores = json.load(f)
else:
    high_scores = {}


def save_torques(torques, data_file_name):
    np.save(os.path.join(GAME_DATA_DIR, f"{data_file_name}.npy"), torques)

# Function to save high scores
def save_high_scores():
    with open(SCORES_FILE, "w") as f:
        json.dump(high_scores, f)

def check_high_score(data_file_name, current_score):
    if data_file_name in high_scores:
        previous_best = high_scores[data_file_name]
        if current_score <= previous_best:
            high_scores[data_file_name] = current_score
            save_high_scores()
            return True  # New high score
    else:
        high_scores[data_file_name] = current_score
        save_high_scores()
        return True
    return False


def end_screen(won, pid_score, current_score, previous_score, next_level_callback, try_again_callback):
    screen.fill(WHITE)
    font = pygame.font.Font(None, 72)

    # Smaller italic font for PID score
    pid_font = pygame.font.Font(None, 48)
    pid_font.set_italic(True)  # Set the font to italic

    # Calculate the score difference (positive if worse, negative if better)
    difference = current_score - previous_score  # Positive means worse, negative means better

    # Determine the color and sign for the difference
    if difference < 0:
        difference_message = f"-{abs(difference):.2f}"  # Negative value means better score
        diff_color = GREEN  # Green for improvement
    else:
        difference_message = f"+{difference:.2f}"  # Positive value means worse score
        diff_color = RED  # Red for worse score

    # Display "You Win" in green or "You Lose" in red
    if won:
        message = "You Win!"
        message_text = font.render(message, True, GREEN)
        score_message = "New High Score"
        score_text = font.render(score_message, True, BLACK)
    else:
        message = "You Lose!"
        message_text = font.render(message, True, RED)
        score_message = f"High Score: {previous_score:.2f}"
        score_text = font.render(score_message, True, BLACK)

    # Combine score and diff in one line
    combined_score_message = f"Score: {current_score:.2f} "
    combined_score_text = font.render(combined_score_message, True, BLACK)
    difference_text = font.render(difference_message, True, diff_color)

    # Display the messages centered
    screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, HEIGHT // 2 - 200))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 60))

    # Display the combined score and difference on the same line
    combined_score_x = WIDTH // 2 - (combined_score_text.get_width() + difference_text.get_width()) // 2
    screen.blit(combined_score_text, (combined_score_x, HEIGHT // 2))
    screen.blit(difference_text, (combined_score_x + combined_score_text.get_width(), HEIGHT // 2))

    # Move PID score lower for better spacing
    pid_score_message = f"PID Score: {pid_score:.2f}"
    pid_score_text = pid_font.render(pid_score_message, True, GRAY)

    # Center the PID score below the score and diff, and give it more space
    screen.blit(pid_score_text, (WIDTH // 2 - pid_score_text.get_width() // 2, HEIGHT // 2 + 60))

    # Button Font
    button_font = pygame.font.Font(None, 50)

    # Draw "Next Level" button with Gray background and White text
    next_level_text = button_font.render("Next Level", True, WHITE)
    next_level_rect = pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 100, HEIGHT // 2 + 150, 200, 50))
    screen.blit(next_level_text, (WIDTH // 2 - next_level_text.get_width() // 2, HEIGHT // 2 + 160))

    # Draw "Try Again" button with Gray background and White text, move it further down
    try_again_text = button_font.render("Try Again", True, WHITE)
    try_again_rect = pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 100, HEIGHT // 2 + 250, 200, 50))
    screen.blit(try_again_text, (WIDTH // 2 - try_again_text.get_width() // 2, HEIGHT // 2 + 260))

    pygame.display.flip()

    # Wait for user input (Next Level or Try Again)
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if next_level_rect.collidepoint(event.pos):
                    next_level_callback()  # Go to next level
                    waiting = False
                elif try_again_rect.collidepoint(event.pos):
                    try_again_callback()  # Retry current level
                    waiting = False
            elif event.type == pygame.QUIT:
                waiting = False


def main():
    global LEVEL_NUM, DATA_PATH
    controller = Controller(LEVEL_NUM)
    pid_model = tinyphysics.TinyPhysicsModel(MODEL_PATH, debug=DEBUG)
    pid_sim = tinyphysics.TinyPhysicsSimulator(pid_model, str(DATA_PATH), controller=pid.Controller(), debug=False)
    pid_sim.rollout()

    sim_model = tinyphysics.TinyPhysicsModel(MODEL_PATH, debug=DEBUG)
    sim = tinyphysics.TinyPhysicsSimulator(sim_model, str(DATA_PATH), controller=controller, debug=DEBUG)
    sim.step_idx = CONTEXT_LENGTH
    running = True
    won = False

    def next_level_callback():
        nonlocal running
        global LEVEL_NUM, DATA_PATH
        running = False
        LEVEL_NUM += 1  # Increment to next level
        DATA_PATH = os.path.join(TINY_DATA_DIR, f"{LEVEL_NUM:05}.csv")
        main()  # Start next level

    def try_again_callback():
        nonlocal running
        running = False
        main()  # Restart the current level

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if sim.step_idx < len(sim.data):
            sim.step()
            if sim.step_idx > 100 and sim.step_idx % 10 == 0:
                score = sim.compute_cost()
                controller.update_score(score)

        else:
            current_score = controller.total_cost
            pid_score = pid_sim.compute_cost().get("total_cost")
            if check_high_score(f"{LEVEL_NUM:05}", current_score):
                won = True
                save_torques(controller.torques, f"{LEVEL_NUM:05}")

            end_screen(won, pid_score, current_score, high_scores.get(f"{LEVEL_NUM:05}", float('inf')), next_level_callback, try_again_callback)
            running = False


if __name__ == "__main__":
    main()
