import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load existing simulation file names
existing_simulations = [sim.split("-")[0] + ".csv" for sim in os.listdir("simulations")]

# Load the single CSV file into a DataFrame
data = pd.read_csv("../../eval-pids-5000.csv")

# Pivot the data to compare controllers
pivoted_data = data.pivot(index='file', columns='controller', values=['jerk_cost', 'lataccel_cost', 'total_cost'])

# Flatten the MultiIndex columns for easier access
pivoted_data.columns = ['_'.join(col) for col in pivoted_data.columns]
pivoted_data.reset_index(inplace=True)

# Indicate if a file is in existing simulations
pivoted_data['in_existing_simulations'] = pivoted_data['file'].isin(existing_simulations)

# Filter files where `pid_model` total_cost is not the lowest
pid_total_cost = "total_cost_pid_model"
other_total_costs = [col for col in pivoted_data.columns if "total_cost_" in col and "pid_model" not in col]

pivoted_data['pid_model_not_lowest'] = pivoted_data[pid_total_cost] > pivoted_data[other_total_costs].min(axis=1)

# Filter for the files where pid_model is not the lowest
worse_files = pivoted_data[pivoted_data['pid_model_not_lowest']].copy()

# Exclude files already in existing simulations
worse_files = worse_files[~worse_files['in_existing_simulations']]

# Calculate the difference between pid_model total_cost and the best-performing controller
worse_files['diff'] = worse_files[pid_total_cost] - worse_files[other_total_costs].min(axis=1)

# Sort by the severity of the difference
worse_files = worse_files.sort_values(by='diff', ascending=False)

# Total sum of diffs and hypothetical score calculations
total_diff_sum = worse_files['diff'].sum()
pid_model_total_sum = pivoted_data[pid_total_cost].sum()
hypothetical_total_sum = pid_model_total_sum - total_diff_sum
hypothetical_average_total_cost = hypothetical_total_sum / len(pivoted_data)

# Print the results
print(f"Sum of all 'diff' values: {total_diff_sum:.3f}")
print(f"Original average total_cost (pid_model): {pid_model_total_sum / len(pivoted_data):.3f}")
print(f"Hypothetical average total_cost (if diffs removed): {hypothetical_average_total_cost:.3f}")

# Adaptive Bins for Frequency Distribution
bin_edges = [1, 10, 20, 30, 40, 50, 60, 80, 100, 500]
worse_files['score_bin'] = pd.cut(worse_files[pid_total_cost], bins=bin_edges, labels=False)
bin_counts = worse_files['score_bin'].value_counts(sort=False)
bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]

# Calculate the impact on the final average total_cost for each bin
impact_per_bin = worse_files.groupby('score_bin')['diff'].sum().sort_index() / len(pivoted_data)  # Impact normalized

# Randomized Candidate Selection
num_candidates = 30
random_seed = 44

# Proportional sampling for candidates based on bin frequency
final_candidates = []
np.random.seed(random_seed)

for bin_idx, count in bin_counts.items():
    # Proportional number of candidates for this bin
    bin_files = worse_files[worse_files['score_bin'] == bin_idx]
    n_samples = max(1, int((count / len(worse_files)) * num_candidates))  # At least 1 sample per bin if it has files
    sampled_files = bin_files.sample(n=min(len(bin_files), n_samples), random_state=random_seed)
    final_candidates.append(sampled_files)

# Combine all sampled files into the final candidates DataFrame
final_candidates_df = pd.concat(final_candidates).reset_index(drop=True)

# Print the final candidates
print(f"Selected {len(final_candidates_df)} candidates for further evaluation:")
print(final_candidates_df[['file', 'diff', 'score_bin']])
print([int(os.path.splitext(file)[0]) for file in final_candidates_df["file"]])

# Save the candidates to a CSV file
final_candidates_df.to_csv('selected_candidates.csv', index=False)

# --- PLOT 1: Adaptive Bin Frequency Plot with Impacts ---
plt.figure(figsize=(12, 6))
bars = plt.bar(bin_labels, bin_counts, color='orange', edgecolor='black', alpha=0.7)

# Add labels showing the impact on the final average total_cost for each bin
for bar, impact in zip(bars, impact_per_bin):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{impact:.5f}", ha='center', fontsize=10, color='black')

plt.title("Frequency of pid_model Worse Scores (Adaptive Bins) with Impact on Final Average Total Cost")
plt.xlabel("Total Cost Range")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- PLOT 2: Normalized Candidate Distribution vs Worse Scores ---
candidate_counts = final_candidates_df['score_bin'].value_counts(sort=False)
candidate_normalized = candidate_counts / candidate_counts.sum()
worse_normalized = bin_counts / bin_counts.sum()

plt.figure(figsize=(12, 6))
bar_width = 0.4
x = np.arange(len(bin_labels))

# Plot original worse scores
plt.bar(x - bar_width / 2, worse_normalized, width=bar_width, label="Worse Scores (Normalized)", color="skyblue", edgecolor="black")

# Plot candidate scores
plt.bar(x + bar_width / 2, candidate_normalized, width=bar_width, label="Candidate Scores (Normalized)", color="orange", edgecolor="black")

# Label adjustments
plt.title("Normalized Comparison of Candidate Distribution vs Worse Scores")
plt.xlabel("Total Cost Range (Bins)")
plt.ylabel("Normalized Frequency")
plt.xticks(x, bin_labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- PLOT 3: Interactive Boxplot with Worse Highlights ---
import plotly.graph_objects as go

fig = go.Figure()

# Add boxplots for each controller
total_cost_columns = [col for col in pivoted_data.columns if "total_cost_" in col]
for col in total_cost_columns:
    controller_name = col.replace("total_cost_", "")
    fig.add_trace(go.Box(
        y=pivoted_data[col],
        name=controller_name,
        boxpoints='outliers',  # Show outliers
        marker=dict(size=4),
        line=dict(width=1)
    ))

# Highlight the worse scores for pid_model in red
fig.add_trace(go.Scatter(
    x=['pid_model'] * len(worse_files),  # Position at pid_model
    y=worse_files[pid_total_cost],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='pid_model worse spots'
))

fig.update_layout(
    title="Interactive Comparison of Total Cost Across Controllers",
    yaxis_title="Total Cost",
    xaxis_title="Controller",
    xaxis=dict(tickangle=45),
    template="plotly_white"
)

fig.show()
