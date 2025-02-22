import tinyphysics
from tinyphysics import CONTEXT_LENGTH
from controllers import BaseController, pid, replay, pid_top, pid_model, experimental
from train.deep.reward import compute_reward
from draw_game import *
import numpy as np

LEVEL_IDX = 5
CHECKPOINT = 0
TINY_DATA_DIR = "../data"
MODEL_PATH = "../models/tinyphysics.onnx"
HISTORY = { "torques": [], "diffs": [] }
DEBUG = True
previous_diff = None


def get_best_torque_sequence(history, block_size=10, lag=3):
    num_attempts = len(history["torques"])
    num_points = len(history["torques"][0])

    # Initialize a list to store the final best torques sequence
    best_torques = []

    # Loop through each block position
    for block_start in range(0, num_points, block_size):
        best_block_torques = None
        best_block_diff_sum = float("inf")

        # Determine the lagged range for the diff block
        diff_block_start = min(block_start + lag, num_points - 1)
        diff_block_end = min(diff_block_start + block_size, num_points)

        # Find the best block across all attempts
        for attempt_idx in range(num_attempts):
            torques = history["torques"][attempt_idx]
            diffs = history["diffs"][attempt_idx]

            # Calculate the sum of absolute diffs for this lagged block
            block_diffs = diffs[diff_block_start:diff_block_end]
            block_diff_sum = sum(abs(d) for d in block_diffs)

            # Determine the corresponding torque block range without lag
            torque_block_start = block_start
            torque_block_end = min(block_start + block_size, num_points)
            selected_torques = torques[torque_block_start:torque_block_end]

            # If this block has the lowest diff sum so far, select it
            if block_diff_sum < best_block_diff_sum:
                best_block_diff_sum = block_diff_sum
                best_block_torques = selected_torques

        # Append the selected block to the final best torques sequence
        best_torques.extend(best_block_torques)

    # Handle any remaining values at the end that don’t fit into a full block
    if len(best_torques) < num_points:
        remaining_torques = history["torques"][0][len(best_torques):num_points]
        best_torques.extend(remaining_torques)

    # Ensure the final output has the exact number of data points by trimming if oversize
    best_torques = best_torques[:num_points]
    return best_torques


class Controller(BaseController):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.internal_pid = experimental.Controller()
        self.internal_replay = replay.Controller(level)
        self.torques = []
        self.lat_accel_cost = 0
        self.jerk_cost = 0
        self.total_cost = 0

        # Initialize torque levels for 1024 steps
        self.increment = 1
        self.ctrl_increment = 2
        self.ctrl_pressed = False
        self.min_torque = -2.5
        self.max_torque = 2.5
        self.num_steps = 256
        self.torque_levels = np.linspace(self.min_torque, self.max_torque, self.num_steps)
        self.current_torque_index = self.num_steps // 2  # Start at zero torque

    def update_score(self, score):
        # Update the stored score values
        self.lat_accel_cost = score['lataccel_cost']
        self.jerk_cost = score['jerk_cost']
        self.total_cost = score['total_cost']

    def map_value_to_range(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def turn_wheel(self, torque_target):
        if pygame.joystick.get_count() > 0:
            while True:
                pygame.event.pump()
                raw_input = -joystick.get_axis(0)
                scaled_input = self.map_value_to_range(raw_input, -1, 1, -2.5, 2.5)
                auto_steer_diff = scaled_input - torque_target
                if abs(auto_steer_diff) <= 0.01:
                    break
                if auto_steer_diff > 0:
                    wheel.force_constant(0.4)
                else:
                    wheel.force_constant(0.6)
            wheel.force_constant(0.5)

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        global BASE_FPS, CHECKPOINT, previous_diff
        pygame.event.pump()

        # Update internal controllers
        future_lataccel = target_lataccel
        if len(future_plan.lataccel) >= 5:
            future_lataccel = future_plan.lataccel[4] # Use future target lataccel (if available)
        pid_action = self.internal_pid.update(future_lataccel, current_lataccel, state, future_plan, steer)
        replay_torque = self.internal_replay.update(target_lataccel, current_lataccel, state, future_plan, steer)

        # Compute custom reward
        reward = 0
        current_diff = target_lataccel - current_lataccel
        if previous_diff:
            reward = compute_reward(abs(previous_diff) - abs(current_diff), current_diff)
        previous_diff = current_diff

        # Determine position on road and SIM index
        index = len(self.torques)

        # # Get the state of all keyboard buttons
        keys = pygame.key.get_pressed()

        # Checkpoint press
        if keys[pygame.K_x] and index > 80:
            CHECKPOINT = index

        if keys[pygame.K_UP]:
            BASE_FPS += 2
        elif keys[pygame.K_DOWN]:
            BASE_FPS -= 2

        # Limit FPS
        BASE_FPS = min(max(BASE_FPS, 2), 60)

        # Calculate FPS based on vego
        if SCALE_FPS:
            FPS = 10.0 * (state.v_ego / 22.0)
        else:
            FPS = BASE_FPS

        # Increment
        if keys[pygame.K_LCTRL]:
            self.ctrl_pressed = True
            self.increment = self.ctrl_increment
        elif self.ctrl_pressed:
            self.ctrl_pressed = False
            self.increment = 1

        if keys[pygame.K_1]:
            self.ctrl_increment = 1
        if keys[pygame.K_2]:
            self.ctrl_increment = 2
        if keys[pygame.K_3]:
            self.ctrl_increment = 3
        if keys[pygame.K_4]:
            self.ctrl_increment = 4
        if keys[pygame.K_5]:
            self.ctrl_increment = 5
        if keys[pygame.K_6]:
            self.ctrl_increment = 6

        if keys[pygame.K_LSHIFT]:
            FPS *= 2

        # Adjust torque index based on key presses
        torque_output = None
        if keys[pygame.K_LEFT]:
            # Decrease torque index (turn left)
            self.current_torque_index = min(max(self.current_torque_index + self.increment, 0), self.num_steps - 1)
        elif keys[pygame.K_RIGHT]:
            # Increase torque index (turn right)
            self.current_torque_index = min(max(self.current_torque_index - self.increment, 0), self.num_steps - 1)
        else:
            # Get raw joystick input [-1, 1]
            if pygame.joystick.get_count() > 0:
                raw_input = -joystick.get_axis(0)
                scaled_input = self.map_value_to_range(raw_input, -1, 1, -2.5, 2.5)
                self.current_torque_index = min(range(len(self.torque_levels)), key=lambda i: abs(self.torque_levels[i] - scaled_input))

        # Use replay data (if key pressed)
        if keys[pygame.K_SPACE] or (CHECKPOINT and index < CHECKPOINT):
            self.internal_pid.correct(replay_torque)
            self.current_torque_index = min(range(len(self.torque_levels)), key=lambda i: abs(self.torque_levels[i] - replay_torque))
            # move wheel to correct position (pause game if needed)
            self.turn_wheel(self.torque_levels[self.current_torque_index])

        if keys[pygame.K_c]:
            self.current_torque_index = min(range(len(self.torque_levels)), key=lambda i: abs(self.torque_levels[i] - pid_action))
            # move wheel to correct position (pause game if needed)
            self.turn_wheel(self.torque_levels[self.current_torque_index])

        # Use initial steer (if any)
        if not math.isnan(steer):
            self.internal_pid.correct(steer)
            self.current_torque_index = min(range(len(self.torque_levels)), key=lambda i: abs(self.torque_levels[i] - steer))
            # move wheel to correct position (pause game if needed)
            self.turn_wheel(self.torque_levels[self.current_torque_index])

        # Get the torque output from the torque levels array
        if not torque_output:
            torque_output = self.torque_levels[self.current_torque_index]

        # Append the attempt's torques and lagged diffs to history
        HISTORY_INDEX = len(HISTORY["torques"]) - 1
        HISTORY["torques"][HISTORY_INDEX].append(torque_output)
        HISTORY["diffs"][HISTORY_INDEX].append(target_lataccel - current_lataccel)

        # Draw the game elements
        draw_background()
        draw_road(future_plan, HEIGHT - 100, current_lataccel, target_lataccel, state.roll_lataccel, index)
        draw_car(WIDTH // 2, HEIGHT - 50, 0, target_lataccel - current_lataccel)
        draw_steering(torque_output, self.ctrl_increment, self.ctrl_pressed)
        draw_speedometer(state.v_ego * 2.23694, 0, 100)
        draw_reward(reward)
        draw_score(self.lat_accel_cost, self.jerk_cost, self.total_cost, FPS)
        draw_level(self.level)
        if keys[pygame.K_SPACE]:
            draw_mode("REPLAY")
        if keys[pygame.K_c]:
            draw_mode("PID")
        if CHECKPOINT and index < CHECKPOINT:
            draw_mode(f"CHECKPOINT {CHECKPOINT-index}")
            FPS *= 4
        elif CHECKPOINT:
            draw_mode(f"CHECKPOINT")

        # Update display
        pygame.display.flip()

        # pause game
        clock.tick(FPS)

        # Return torque value to simulator (not used in drawing)
        self.torques.append(torque_output)
        return torque_output


def main():
    global LEVEL_IDX
    LEVEL_NUM = LEVELS[LEVEL_IDX]

    # Add new history record
    HISTORY["torques"].append([])
    HISTORY["diffs"].append([])

    DATA_PATH = os.path.join(TINY_DATA_DIR, f"{LEVEL_NUM:05}.csv")
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
        global LEVEL_IDX, DATA_PATH, CHECKPOINT, HISTORY
        running = False
        CHECKPOINT = 0
        LEVEL_IDX += 1  # Increment to next level
        LEVEL_NUM = LEVELS[LEVEL_IDX]
        HISTORY = {"torques": [], "diffs": []}
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
            if sim.step_idx > 100 and sim.step_idx % 10 == 0 or sim.step_idx == len(sim.data) - 1:
                score = sim.compute_cost()
                controller.update_score(score)

        else:
            current_score = controller.total_cost
            pid_score = pid_sim.compute_cost().get("total_cost")
            previous_score = high_scores.get(f"{LEVEL_NUM:05}", 0.0)

            # Get best blocks from history
            best_combined_torques = list(get_best_torque_sequence(HISTORY))

            # Rollout of 'best' score to verify
            pid_model = tinyphysics.TinyPhysicsModel(MODEL_PATH, debug=DEBUG)
            replay_sim = tinyphysics.TinyPhysicsSimulator(pid_model, str(DATA_PATH), controller=replay.Controller(torques=best_combined_torques), debug=False)
            replay_sim.rollout()
            best_combined_score = replay_sim.compute_cost()["total_cost"]

            if best_combined_score < current_score:
                print(f"Best score {best_combined_score} lower than current score: {current_score}, replacing it...")
                current_score = best_combined_score
            else:
                print(f"Combined score is {best_combined_score}")

            if check_high_score(f"{LEVEL_NUM:05}", current_score):
                print(f"New high score: {current_score}, replacing previous high score: {previous_score}")
                won = True
                save_torques(controller.torques, f"{LEVEL_NUM:05}")

            end_screen(won, pid_score, current_score, previous_score, next_level_callback, try_again_callback)
            running = False


if __name__ == "__main__":
    main()
