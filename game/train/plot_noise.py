import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Time steps for simulation
T = 100
t = np.arange(T)

# 1. Pure Random Noise: each step is independent
pure_random = np.random.uniform(-10, 10, T)

# 2. Ornstein-Uhlenbeck Noise
def generate_ou_noise(T, mu=0, theta=0.15, sigma=1, dt=1):
    x = np.zeros(T)
    x[0] = 0
    for i in range(1, T):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x

ou_noise = generate_ou_noise(T)

# 3. Linear Interpolation Noise:
# Every 10 steps, choose a new target offset in range [-10, 10], and move linearly toward it
linear_noise = np.zeros(T)
target = 0
step = 0
for i in range(T):
    if i % 10 == 0:
        target = np.random.uniform(-10, 10)
        step = 0
    # Calculate linear increment to move toward target in 10 steps
    increment = (target - linear_noise[i-1]) / (10 - step) if i > 0 else 0
    linear_noise[i] = linear_noise[i-1] + increment if i > 0 else 0
    step += 1

# 4. Low-Pass Filtered Gaussian Noise (Exponential Moving Average)
gaussian_noise = np.random.normal(0, 2, T)
alpha = 0.1  # smoothing factor
ema_noise = np.zeros(T)
ema_noise[0] = gaussian_noise[0]
for i in range(1, T):
    ema_noise[i] = alpha * gaussian_noise[i] + (1 - alpha) * ema_noise[i-1]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, np.zeros(T), 'k--', label="Ideal (0)")  # ideal straight line

plt.plot(t, pure_random, label="Pure Random Noise", alpha=0.7)
plt.plot(t, ou_noise, label="Ornstein-Uhlenbeck Noise", alpha=0.7)
plt.plot(t, linear_noise, label="Linear Interpolation Noise", alpha=0.7)
plt.plot(t, ema_noise, label="Low-Pass Filtered Gaussian", alpha=0.7)

plt.xlabel("Time step")
plt.ylabel("Steering Value (noise added)")
plt.title("Noise Strategies on a Straight Driving Controller (Ideal = 0)")
plt.legend()
plt.grid(True)
plt.show()
