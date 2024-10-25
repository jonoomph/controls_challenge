import pygame
import json
import os
import numpy as np
from g29py import g29

# Initialize PyGame and the PS5 controller
pygame.init()
pygame.joystick.init()

# Set up the clock to control the FPS
clock = pygame.time.Clock()
BASE_FPS = 8
SCALE_FPS = False
GAME_DATA_DIR = "data"
LEVELS_FILE = os.path.join(GAME_DATA_DIR, "levels.json")
SCORES_FILE = os.path.join(GAME_DATA_DIR, "high_scores.json")

# Assuming there is only one joystick (your PS5 controller)
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    wheel = g29.G29()
    wheel.autocenter_off()
    wheel.set_range(900)
    wheel.set_friction(0.1)

# Set up the screen dimensions and colors
WIDTH, HEIGHT = 1280, 1080  # Tall window size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
TEAL = (0, 40, 100)

# Init screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Simulation")

# Car constants
CAR_WIDTH, CAR_HEIGHT = 30, 50
ROAD_WIDTH = 1280
SEGMENT_HEIGHT = 40
MAX_LATACCEL_DIFF = 5
ROAD_AGGRESSIVE_FACTOR = 160

# Load images
background_image = pygame.image.load("images/background.png").convert()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

car_red_image = pygame.image.load("images/car-red.png").convert_alpha()
car_red_image = pygame.transform.scale(car_red_image, (CAR_WIDTH, CAR_HEIGHT))
car_orange_image = pygame.image.load("images/car-orange.png").convert_alpha()
car_orange_image = pygame.transform.scale(car_orange_image, (CAR_WIDTH, CAR_HEIGHT))
car_green_image = pygame.image.load("images/car-green.png").convert_alpha()
car_green_image = pygame.transform.scale(car_green_image, (CAR_WIDTH, CAR_HEIGHT))

road1_image = pygame.image.load("images/road.png").convert_alpha()
road1_image = pygame.transform.scale(road1_image, (ROAD_WIDTH, SEGMENT_HEIGHT))
road2_image = pygame.image.load("images/road-1.png").convert_alpha()
road2_image = pygame.transform.scale(road2_image, (ROAD_WIDTH, SEGMENT_HEIGHT))
road3_image = pygame.image.load("images/road-2.png").convert_alpha()
road3_image = pygame.transform.scale(road3_image, (ROAD_WIDTH, SEGMENT_HEIGHT))
road4_image = pygame.image.load("images/road-3.png").convert_alpha()
road4_image = pygame.transform.scale(road4_image, (ROAD_WIDTH, SEGMENT_HEIGHT))
road_checker_image = pygame.image.load("images/road-checker.png").convert_alpha()
road_checker_image = pygame.transform.scale(road_checker_image, (ROAD_WIDTH, SEGMENT_HEIGHT))

wheel_image = pygame.image.load("images/wheel.png").convert_alpha()

blank_image = pygame.Surface((1, 1))
blank_image.fill((0, 0, 0))
blank_image.set_colorkey((0, 0, 0))

# Initialize font for displaying score
font = pygame.font.Font(None, 36)
level_font = pygame.font.Font(None, 50)

# Function to draw a green cross above the car with a specified height
def draw_cross_above_car(car_x, car_y, height_above_car, color):
    cross_size = 25  # The length of each arm of the cross
    cross_thickness = 4  # The thickness of the cross lines

    # Coordinates for the vertical line of the cross (height_above_car pixels above the car)
    vertical_start = (car_x, car_y - height_above_car - cross_size // 2)
    vertical_end = (car_x, car_y - height_above_car + cross_size // 2)

    # Coordinates for the horizontal line of the cross
    horizontal_start = (car_x - cross_size // 2, car_y - height_above_car)
    horizontal_end = (car_x + cross_size // 2, car_y - height_above_car)

    # Draw the vertical line of the cross
    pygame.draw.line(screen, color, vertical_start, vertical_end, cross_thickness)

    # Draw the horizontal line of the cross
    pygame.draw.line(screen, color, horizontal_start, horizontal_end, cross_thickness)

# Function to draw the car
def draw_car(car_x, car_y, rotation_angle, distance_from_target):
    # Call the function to draw the cross 20 pixels above the car
    if abs(distance_from_target) < 0.1:
        car_image = car_green_image
    elif abs(distance_from_target) < 0.3:
        car_image = car_orange_image
    else:
        car_image = car_red_image
    screen.blit(car_image, car_image.get_rect(center=(car_x, car_y)))
    #draw_cross_above_car(car_x, car_y, 0, color)

def get_road_segment_image(index):
    if index in [80, 577]:
        return road_checker_image
    if index < 144:
        return road1_image
    elif index < 288:
        return road2_image
    elif index < 433:
        return road3_image
    elif index < 577:
        return road4_image
    else:
        return blank_image

def draw_background():
    screen.blit(background_image, (0, 0))

def draw_road(future_plan, car_y, current_lataccel, target_lataccel, roll_lataccel, index):
    # Start by drawing the furthest segment first
    num_segments = 41  # Number of segments to draw
    min_scale = 0.3  # Minimum scale for the furthest segment
    max_scale = 1.0  # Maximum scale for the closest segment

    # Create a list of lateral accelerations (reversed future segments + current segment)
    lataccels = [target_lataccel] + list(future_plan.lataccel[:num_segments])
    if len(lataccels) < num_segments:
        padding_needed = num_segments - len(lataccels)
        lataccels.extend([0.0] * padding_needed)

    total_height = 0  # Track total height to correctly position Y-axis

    # Iterate through the lateral accelerations list
    for i in reversed(range(num_segments)):
        lataccel_diff = current_lataccel - lataccels[i]

        # Calculate the correct scale factor: smallest for furthest, largest for closest
        scale_factor = min_scale + (max_scale - min_scale) * ((num_segments - 1 - i) / (num_segments - 1))

        # Calculate the width and height of the current segment
        scaled_road_width = int(ROAD_WIDTH * scale_factor)
        scaled_road_height = int(SEGMENT_HEIGHT * scale_factor)

        # Calculate the vertical position of the segment
        segment_y = total_height  # Start from the top of the screen (0) and increment downwards
        total_height += scaled_road_height  # Accumulate height downwards to fill the screen

        # Calculate the horizontal center of the road segment based on lateral acceleration
        road_center_x = (WIDTH // 2) + (lataccel_diff * ROAD_AGGRESSIVE_FACTOR)
        road_x = road_center_x - (scaled_road_width // 2)

        # Scale and draw the road segment
        scaled_road = pygame.transform.scale(get_road_segment_image(i + index), (scaled_road_width, scaled_road_height))
        screen.blit(scaled_road, (road_x, segment_y))

        # Debug print
        print(f"Segment {i}: Scale: {scale_factor:.2f}, Width: {scaled_road_width}, Height: {scaled_road_height}, X: {road_x}, Y: {segment_y}")

    print("--- END DEBUG INFO ---\n")  # End of debug information

def draw_steering(torque_value, increment, ctrl_pressed):
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

    increment_color = WHITE
    if ctrl_pressed:
        increment_color = GREEN
    increment_text = font.render(f"+{increment}", True, increment_color)

    # Draw the torque text in the middle of the wheel (unrotated)
    screen.blit(torque_text, (text_rect.x, text_rect.y + 5))
    screen.blit(increment_text, (text_rect.x + 15, text_rect.y - 20))

def draw_arrow(road_x, road_y, road_roll):
    # Constants
    max_arrow_length = 6000  # Max arrow length in pixels, used to scale road roll
    arrow_height = 2         # Height of the arrow line in pixels
    triangle_size = 10       # Size of the triangle at the end of the arrow
    threshold = 15.0         # Threshold below which we don't show the triangle

    # Calculate the length of the arrow based on road_roll magnitude
    arrow_length = max_arrow_length * abs(road_roll)
    road_center = ROAD_WIDTH // 2

    # Define the side of the road where the arrow should be drawn
    if road_roll > 0:
        start_x = road_x + road_center - 120  # Left side of the road
        end_x = start_x - arrow_length  # Arrow points left
    else:
        start_x = road_x + road_center + 120  # Right side of the road
        end_x = start_x + arrow_length  # Arrow points right

    start_y = road_y

    # Draw the triangle if abs(road_roll) is above the threshold
    if abs(arrow_length) >= threshold:
        # Draw the horizontal line (2px tall)
        pygame.draw.line(screen, WHITE, (start_x, start_y), (end_x, start_y), arrow_height)

        # Triangle points to the direction of the roll
        if road_roll > 0:
            triangle_points = [(end_x, start_y),
                               (end_x + triangle_size, start_y - triangle_size // 2),
                               (end_x + triangle_size, start_y + triangle_size // 2)]
        else:
            triangle_points = [(end_x, start_y),
                               (end_x - triangle_size, start_y - triangle_size // 2),
                               (end_x - triangle_size, start_y + triangle_size // 2)]

        # Draw the triangle
        pygame.draw.polygon(screen, WHITE, triangle_points)


def draw_score(lat_accel_cost, jerk_cost, total_cost, FPS):
    # Create the text surfaces
    lat_text = font.render(f"Lat: {lat_accel_cost:.2f}", True, WHITE)
    jerk_text = font.render(f"Jerk: {jerk_cost:.2f}", True, WHITE)
    total_text = font.render(f"Total: {total_cost:.2f}", True, WHITE)
    fps_text = font.render(f"FPS: {FPS}", True, WHITE)

    # Position the text at the top of the screen
    screen.blit(lat_text, (20, 20))
    screen.blit(jerk_text, (20, 60))
    screen.blit(total_text, (20, 100))
    screen.blit(fps_text, (20, 140))

def draw_mode(mode_name="REPLAY"):
    # Create the text surface
    replay_text = font.render(mode_name, True, WHITE, BLACK)

    # Get the width and height of the text surface
    text_width, text_height = replay_text.get_size()

    # Position the text in the top-right corner of the screen
    x_position = WIDTH - text_width - 25  # 20px padding from the right
    y_position = 60  # 20px padding from the top

    # Draw the text with the background
    screen.blit(replay_text, (x_position , y_position))

def draw_level(level_num):
    # Create the text surface for the level number
    level_text = level_font.render(f"Level: {level_num}", True, WHITE)

    # Position the level text at the top-right of the screen
    screen.blit(level_text, (WIDTH - level_text.get_width() - 20, 20))


def end_screen(won, pid_score, current_score, previous_score, next_level_callback, try_again_callback):
    screen.fill(WHITE)
    font = pygame.font.Font(None, 72)

    # Smaller italic font for PID score
    pid_font = pygame.font.Font(None, 48)
    pid_font.set_italic(True)  # Set the font to italic

    # Calculate the score difference (positive if worse, negative if better)
    difference = current_score - previous_score  # Positive means worse, negative means better

    # Determine the color and sign for the difference
    if difference <= 0:
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


# Load high scores or initialize if it doesn't exist
if os.path.exists(SCORES_FILE):
    with open(SCORES_FILE, "r") as f:
        high_scores = json.load(f)
else:
    high_scores = {}

# Load levels file
with open(LEVELS_FILE, "r") as f:
    LEVELS = json.load(f)

def save_torques(torques, data_file_name):
    np.save(os.path.join(GAME_DATA_DIR, f"{data_file_name}.npy"), torques)

# Function to save high scores
def save_high_scores():
    with open(SCORES_FILE, "w") as f:
        json.dump({key: high_scores[key] for key in sorted(high_scores)}, f, indent=4)

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
