import onnxruntime as ort
import numpy as np
from . import BaseController
from . import experimental
import math


class Controller(BaseController):
    """
    AI-powered PID controller using an ONNX model for prediction. This controller also requires the initial steer
    values at the beginning of the simulation to prepare the model for predictions.
    """

    def __init__(self, window_size=30, model_path="/home/jonathan/apps/controls_challenge/game/train/onnx/model-jAXpb-17.onnx"):
        """
        Initialize the controller with the ONNX model and time-series window parameters.

        Args:
            window_size (int): Size of the time-series window for model input.
            model_path (str): Path to the ONNX model file.
        """
        # Set up the ONNX runtime session
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = [
            #'CUDAExecutionProvider',
            'CPUExecutionProvider']

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, providers)

        # Controller parameters
        self.window_size = window_size
        self.input_window = []  # Sliding window for input data
        self.prev_actions = []  # Store previous control signals
        self.step_idx = 20  # Simulation step index
        self.internal_pid = experimental.Controller()

        # Buffers for median filtering (optional future use)
        self.steer_window = []
        self.lataccel_window = []

        # Predefined torque levels
        self.torques = np.linspace(-2.5, 2.5, 256)

    def average(self, values):
        """ Calculate the average of a list of values, handling empty lists gracefully. """
        return sum(values) / len(values) if values else 0

    def normalize_v_ego(self, v_ego_m_s):
        """
        Normalize vehicle speed to a range [0, 1] using square root scaling.

        Args:
            v_ego_m_s (float): Vehicle speed in meters per second.

        Returns:
            float: Normalized speed.
        """
        max_m_s = 40.0  # Maximum speed for normalization
        v_ego_m_s = max(0, v_ego_m_s)  # Clamp negative speeds to zero
        return math.sqrt(v_ego_m_s) / math.sqrt(max_m_s)

    def update(self, target_lataccel, current_lataccel, state, future_plan, steer):
        """
        Update the control signal using the ONNX model and traditional PID logic.

        Args:
            target_lataccel (float): Target lateral acceleration.
            current_lataccel (float): Current lateral acceleration.
            state (object): Current vehicle state containing roll, speed, and acceleration.
            future_plan (object): Predicted future states for the vehicle.
            steer (float): Override steering value (if not NaN).

        Returns:
            float: Control signal for steering.
        """
        # Compute initial PID control signal (using Experimental PID)
        pid_steer = self.internal_pid.update(target_lataccel, current_lataccel, state, future_plan, math.nan)

        # Compute differences for future segments
        future_segments = [(1, 2), (2, 3), (3, 4)]
        diff_values = {
            'lataccel': [current_lataccel - self.average(future_plan.lataccel[start:end]) for start, end in future_segments],
            'roll': [state.roll_lataccel - self.average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
            'v_ego': [self.normalize_v_ego(self.average(future_plan.v_ego[start:end])) for start, end in future_segments],
            'a_ego': [self.average(future_plan.a_ego[start:end]) for start, end in future_segments],
        }

        # Include previous steering torques in the input
        previous_action = self.prev_actions[-3:] if len(self.prev_actions) >= 3 else [0, 0, 0]

        num_candidates = 15
        torque_idx = np.argmin(np.abs(self.torques - pid_steer))
        half_range = (num_candidates - 1) // 2

        # Determine candidate indices using slicing with max/min to handle edges
        candidate_indices = [
            max(0, min(len(self.torques) - 1, torque_idx + offset))
            for offset in range(-half_range, half_range + 1)
        ]

        # Extract candidate torques
        candidate_torques = list(self.torques[candidate_indices])
        #print(candidate_torques)

        # Create input state vector for the model
        state_input = np.array(
            diff_values['lataccel'] +
            diff_values['roll'] +
            diff_values['a_ego'] +
            diff_values['v_ego'] +
            previous_action +
            candidate_torques,
            dtype=np.float32
        )

        # Update sliding window
        self.input_window.append(state_input)

        # Predict control signal if enough data is available
        control_signal = 0
        if len(self.input_window) >= self.window_size:
            input_tensor = np.array(self.input_window[-self.window_size:]).reshape(1, self.window_size, -1)
            output = self.ort_session.run(None, {'input': input_tensor})[0]
            torque_scores = output[0]

            # Find the index of the lowest score
            best_index = np.argmin(torque_scores)
            print(torque_scores)
            control_signal = candidate_torques[best_index]

        # Use manual steer value if provided
        if not math.isnan(steer):
            control_signal = steer

        # Save control signal and increment step
        self.prev_actions.append(control_signal)
        self.step_idx += 1

        return control_signal
