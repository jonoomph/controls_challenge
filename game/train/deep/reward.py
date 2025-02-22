import math
import numpy as np


class RewardNormalizer:
    def __init__(self, rewards):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        self.var += (batch_var - self.var) * batch_count / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, rewards):
        return (rewards - self.mean) / np.sqrt(self.var + 1e-8)


def compute_reward(improvement, current_error, k_proximity=4.0, improvement_scale=1.2, penalty_scale=2.0):
    """
    Compute reward scaled between -1 (worst) and 1 (best).

    Args:
        improvement (float): Previous error minus current error (positive = improving).
        current_error (float): Current absolute error from the target.
        k_proximity (float): Controls proximity reward decay rate.
        improvement_scale (float): Reward multiplier for improvement.
        penalty_scale (float): Penalty multiplier for worsening.
    """
    # Proximity reward: [0, 1] (1 = at target, 0 = far)
    proximity_reward = math.exp(-k_proximity * abs(current_error))

    # Movement reward
    if improvement > 0:
        movement_reward = improvement_scale * improvement  # Reward improvement
    else:
        movement_reward = penalty_scale * improvement  # Penalize worsening

    # Combine and clip to [-1, 1]
    total_reward = proximity_reward + movement_reward
    return max(min(total_reward, 1.0), -1.0)

# Revised test scenarios with correct improvement signs
scenarios = [
    # (description, improvement, current_error)
    ("Worst: Far + Moving Away", -0.5, 2.0),    # Reward = 0.0 + (2.0 * -0.5) = -1.0
    ("Best: On Target", 0.0, 0.0),              # Reward = 1.0 + 0 = 1.0
    ("Far + Strong Improvement", 0.8, 1.5),      # Reward ~0.0 + (1.2*0.8) = 0.96 → 0.96
    ("Near + Slight Worsening", -0.1, 0.1),      # Reward ~0.67 + (2.0*-0.1) = 0.47
    ("Moderate + No Change", 0.0, 0.5),          # Reward ~0.13 + 0 = 0.13
]

print("Scenario reward calculations:")
for label, improvement, current_error in scenarios:
    reward = compute_reward(improvement, current_error)
    print(f"{label:45s} | Improv: {improvement:5.2f}, Error: {current_error:5.2f}, Reward: {reward:.4f}")