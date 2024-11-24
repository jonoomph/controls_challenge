import pandas as pd
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to your training directory and evaluation results
training_dir = './simulations'
training_files = [f"{os.path.splitext(file)[0].split('-')[0]}" for file in os.listdir(training_dir)]
eval_results_path = '/home/jonathan/apps/controls_challenge/eval-51-521-score.csv'

# Load evaluation results
eval_df = pd.read_csv(eval_results_path)
eval_df['file'] = eval_df['file'].str.replace('.csv', '', regex=False)
test_eval_df = eval_df[eval_df['controller'] == 'test']

# Path to simulation data directory
data_dir = '../../data'
drives_summary = []

# Iterate over all CSV files in the directory
for filename in sorted(os.listdir(data_dir))[:5000]:
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path).head(600)

        # Get score details for file
        score = test_eval_df.loc[test_eval_df["file"] == os.path.splitext(filename)[0]]["total_cost"].values[0]

        # Compute statistics for each drive
        drive_summary = {
            'file': os.path.splitext(filename)[0],
            'vEgo_mean': df['vEgo'].mean(),
            'vEgo_std': df['vEgo'].std(),
            'aEgo_mean': df['aEgo'].mean(),
            'aEgo_std': df['aEgo'].std(),
            'targetLateralAcceleration_mean': df['targetLateralAcceleration'].mean(),
            'targetLateralAcceleration_std': df['targetLateralAcceleration'].std(),
            'score': score
        }

        # For steerCommand, compute statistics only for the first 80 rows
        steerCommand_80 = df['steerCommand'].head(80)
        drive_summary['steerCommand_mean'] = steerCommand_80.mean()
        drive_summary['steerCommand_std'] = steerCommand_80.std()

        # Append the summary for this drive
        drives_summary.append(drive_summary)

# Convert the summary list to a DataFrame for further analysis
summary_df = pd.DataFrame(drives_summary)

# Normalize and cluster (excluding 'score' for clustering)
scaler = StandardScaler()
summary_df_scaled = scaler.fit_transform(summary_df[['vEgo_mean', 'vEgo_std', 'aEgo_mean', 'aEgo_std',
                                                     'targetLateralAcceleration_mean', 'targetLateralAcceleration_std',
                                                     'steerCommand_mean', 'steerCommand_std']])

# Use k-means clustering to cluster the drives into groups
kmeans = KMeans(n_clusters=100, random_state=44)
summary_df['cluster'] = kmeans.fit_predict(summary_df_scaled)

# Merge with evaluation results to analyze performance
merged_df = pd.merge(summary_df, test_eval_df, how='left', left_on='file', right_on='file')
merged_df['is_training'] = merged_df['file'].apply(lambda x: 1 if x in training_files else 0)
filtered_df = merged_df.loc[merged_df['is_training'] == 0]

# Define score range for candidates
min_score = 10
max_score = 150
num_candidates = 80  # Total number of candidates

# Filter out files with scores outside the range
filtered_df = filtered_df[(filtered_df['total_cost'] >= min_score) & (filtered_df['total_cost'] <= max_score)]

# Define adaptive bins based on data distribution
# Small bins for dense areas and large bins for sparse areas
bin_edges = [10, 20, 30, 40, 50, 60, 80, 100, 150]  # Manually defined adaptive bins
filtered_df['score_bin'] = pd.cut(filtered_df['total_cost'], bins=bin_edges, labels=False)

# Compute the mean score for each bin and add it as a label
bin_means = filtered_df.groupby('score_bin')['total_cost'].mean().sort_index()
filtered_df['bin_mean_score'] = filtered_df['score_bin'].map(bin_means)

# Stratified sampling: Match the original distribution of score bins
bin_counts = filtered_df['score_bin'].value_counts(normalize=True).sort_index()
final_candidates = []

for bin_idx, bin_percentage in bin_counts.items():
    bin_files = filtered_df[filtered_df['score_bin'] == bin_idx]
    clusters_in_bin = bin_files['cluster'].unique()

    # Select one file per cluster in this bin, maximizing unique clusters
    candidates = []
    for cluster in clusters_in_bin:
        cluster_files = bin_files[bin_files['cluster'] == cluster]
        candidates.append(cluster_files.sample(n=1, random_state=44))  # Randomly sample one file per cluster

    # Combine cluster-based candidates and adjust total sample size per bin
    bin_candidates = pd.concat(candidates)
    n_samples = max(1, int(bin_percentage * num_candidates))  # Adjust total candidates
    final_candidates.append(bin_candidates.sample(n=min(len(bin_candidates), n_samples), random_state=44))

# Combine all sampled files into the final candidates DataFrame
final_candidates_df = pd.concat(final_candidates).reset_index(drop=True)

# Sort final candidates by score for output
sorted_candidates = final_candidates_df[['file', 'total_cost']].sort_values(by='total_cost', ascending=False)
print("List of candidate files with their current model scores (sorted by highest score):")
print(sorted_candidates)
print([int(row) for row in sorted_candidates['file']])

# Save sorted list of candidate files and scores to a CSV file for easy reference
sorted_candidates.to_csv('../data/candidate_files_with_scores.csv', index=False)

# Extract just the file numbers without '.csv' extension for JSON output
candidate_files = sorted_candidates['file'].str.replace('.csv', '', regex=False).astype(int).tolist()
with open('../data/missing-levels.json', 'w') as json_file:
    json.dump(candidate_files, json_file)

print(f'Successfully saved {len(candidate_files)} candidate file numbers to missing-levels.json')

# Visualization: Updated with Adaptive Bins
plt.figure(figsize=(12, 6))
bin_centers = [np.mean([bin_edges[i], bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]

# Normalize both histograms by dividing counts by their total
original_counts, _ = np.histogram(filtered_df['total_cost'], bins=bin_edges)
candidate_counts, _ = np.histogram(final_candidates_df['total_cost'], bins=bin_edges)

original_counts_normalized = original_counts / original_counts.sum()
candidate_counts_normalized = candidate_counts / candidate_counts.sum()

# Plot normalized total cost scores
plt.bar(bin_centers, original_counts_normalized, width=np.diff(bin_edges), align='center',
        alpha=0.7, label='Total Cost Score', color='skyblue', edgecolor='black')

# Plot normalized candidate scores
plt.bar(bin_centers, candidate_counts_normalized, width=np.diff(bin_edges), align='center',
        alpha=0.7, label='Candidate Scores', color='orange', edgecolor='black')

# Add mean and median lines for the original distribution
plt.axvline(filtered_df['total_cost'].mean(), color='red', linestyle='dashed', linewidth=1,
            label=f'Mean (Total): {filtered_df['total_cost'].mean():.2f}')
plt.axvline(filtered_df['total_cost'].median(), color='green', linestyle='dashed', linewidth=1,
            label=f'Median (Total): {filtered_df['total_cost'].median():.2f}')

# Add labels for each bin's mean score
for i, bin_mean in enumerate(bin_means):
    plt.text(bin_centers[i], max(original_counts_normalized[i], candidate_counts_normalized[i]) + 0.01,
             f'{bin_mean:.1f}', ha='center', fontsize=8)

# Title and labels
plt.title('Normalized Comparison of Total Cost Scores and Candidate Scores (Adaptive Bins)')
plt.xlabel('Mean Score of Each Bin')
plt.ylabel('Proportion (Normalized)')
plt.legend()
plt.show()
