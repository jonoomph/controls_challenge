# config.py
# Hyperparameters for deep Q-learning training

# Training parameters
epochs = 100
batch_size = 44
learning_rate = 1e-5
gamma = 0.99      # Discount factor
n_step = 3        # Use 3-step returns (you can try 5)

# Exploration (epsilon) parameters
initial_epsilon = 1.0
final_epsilon = 0.1
epsilon_decay_epochs = 50  # Epsilon decays linearly over the first 10 epochs

# Noise and action parameters
max_noise = 0.02
action_space = 257         # Discrete actions from -128 to 128
steering_torque_range = (-2.5, 2.5)

# Environment and simulation parameters
num_sims = 50
num_mini_batches = 10
replay_buffer_size =  25000

# Model architecture parameters
window_size = 30
input_size = 15
hidden_size = 80
num_layers = 2

# Miscellaneous
seed = 2002
export_interval = 1
