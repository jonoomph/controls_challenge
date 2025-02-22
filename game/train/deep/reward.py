import math


def compute_reward(improvement, current_error, k_proximity=3.0, improvement_scale=2.5, penalty_scale=1.5):
    """
    Compute reward based on proximity to target and improvement direction/speed.

    Args:
        improvement (float): Previous error minus current error (positive = improving).
        current_error (float): Current absolute error from the target.
        k_proximity (float): Controls how fast proximity reward decays with error.
        improvement_scale (float): Reward multiplier for improvement speed.
        penalty_scale (float): Penalty multiplier for worsening.
    """
    proximity_reward = math.exp(-k_proximity * abs(current_error))

    if improvement > 0:  # Moving toward target
        movement_reward = improvement_scale * improvement
    else:  # Moving away or no change
        movement_reward = penalty_scale * improvement  # improvement is negative/zero

    total_reward = proximity_reward + movement_reward
    return max(min(total_reward, 1.0), 0.0)  # Clipped to [0, 1]


# Revised test scenarios with correct improvement signs
scenarios = [
    # (description, improvement, current_error)
    ("Perfect (on target)", 0.0, 0.0),
    ("Slight positive near target (improving)", 0.02, 0.015),
    ("Slight negative near target (improving)", 0.02, -0.015),
    ("Improving moderate positive", 0.1, 0.3),
    ("Worsening moderate positive", -0.1, 0.3),
    ("Improving far positive", 0.3, 1.0),
    ("Worsening far positive", -0.2, 1.0),
    ("Stable near target", 0.0, 0.01),
    ("Stable far from target", 0.0, 0.8),
]

print("Scenario reward calculations:")
for label, improvement, current_error in scenarios:
    reward = compute_reward(improvement, current_error)
    print(f"{label:45s} | Improv: {improvement:5.2f}, Error: {current_error:5.2f}, Reward: {reward:.4f}")