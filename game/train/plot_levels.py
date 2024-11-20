import pandas as pd
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to your training directory and evaluation results
training_dir = '../data-optimized'
training_files = [f"{os.path.splitext(file)[0].split('-')[0]}" for file in os.listdir(training_dir)]
eval_results_path = '/home/jonathan/apps/controls_challenge/eval-52-201-score.csv'

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

        # Compute statistics for each drive
        drive_summary = {
            'file': os.path.splitext(filename)[0],
            'vEgo_mean': df['vEgo'].mean(),
            'vEgo_std': df['vEgo'].std(),
            'aEgo_mean': df['aEgo'].mean(),
            'aEgo_std': df['aEgo'].std(),
            'targetLateralAcceleration_mean': df['targetLateralAcceleration'].mean(),
            'targetLateralAcceleration_std': df['targetLateralAcceleration'].std(),
        }

        # For steerCommand, compute statistics only for the first 80 rows
        steerCommand_80 = df['steerCommand'].head(80)
        drive_summary['steerCommand_mean'] = steerCommand_80.mean()
        drive_summary['steerCommand_std'] = steerCommand_80.std()

        # Append the summary for this drive
        drives_summary.append(drive_summary)

# Convert the summary list to a DataFrame for further analysis
summary_df = pd.DataFrame(drives_summary)

# Normalize and cluster
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
# [(file, file in training_files) for file in merged_df['file'] ]

# Identify underrepresented clusters (those not fully represented in training data)
underrepresented_clusters = merged_df[merged_df['is_training'] == 0]['cluster'].unique()

# Filter the merged_df for candidate selection: candidates above the mean score and capped at 300
mean_score = 30
score_cap = 600
filtered_candidates = merged_df[
    #(merged_df['total_cost'] > mean_score) &
    (merged_df['total_cost'] <= score_cap) &
    (merged_df['cluster'].isin(underrepresented_clusters))
    ]

# Select top candidates from each underrepresented cluster, based on highest scores up to the score cap
candidates_per_cluster = 1  # Number of top candidates to select per cluster; adjust as needed
final_candidates_df = filtered_candidates.groupby('cluster').apply(
    lambda x: x.nlargest(candidates_per_cluster, 'total_cost')
).reset_index(drop=True)

# Limit the final number of candidates to around 25, if necessary
num_final_candidates = 30
if len(final_candidates_df) > num_final_candidates:
    final_candidates_df = final_candidates_df.nlargest(num_final_candidates, 'total_cost')

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

# ---------- Visualization Section ------------------

# Add is_candidate flag for visualization
pca_df = pd.DataFrame(data=PCA(n_components=2).fit_transform(summary_df_scaled), columns=['PCA1', 'PCA2'])
pca_df['cluster'] = summary_df['cluster']
pca_df['total_cost'] = merged_df['total_cost']
pca_df['is_training'] = merged_df['is_training']
pca_df['file'] = merged_df['file']
pca_df['total_cost_capped'] = np.minimum(pca_df['total_cost'], score_cap)  # Capping for visualization
pca_df['is_candidate'] = pca_df['file'].isin(final_candidates_df['file'])

# Plotting the distribution with capped total_cost for color intensity
plt.figure(figsize=(10, 7))
norm = plt.Normalize(vmin=0, vmax=score_cap)
scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['total_cost_capped'], cmap='viridis', s=10, alpha=0.5,
                      norm=norm)
plt.colorbar(scatter, label=f'Total Cost (Capped at {score_cap})')
plt.title('PCA of Simulation Data with Model Performance (Capped)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Highlight current training files in red
plt.scatter(pca_df[pca_df['is_training'] == 1]['PCA1'],
            pca_df[pca_df['is_training'] == 1]['PCA2'],
            color='red', s=50, label="Current Training Files", edgecolor='black')

# Highlight new candidate files in yellow
plt.scatter(pca_df[pca_df['is_candidate'] == 1]['PCA1'],
            pca_df[pca_df['is_candidate'] == 1]['PCA2'],
            color='yellow', s=50, label="New Candidate Files", edgecolor='black')

plt.legend()
plt.show()


# Basic statistics for the total_cost scores
mean_score = merged_df['total_cost'].mean()
median_score = merged_df['total_cost'].median()
min_score = merged_df['total_cost'].min()
max_score = merged_df['total_cost'].max()
std_dev = merged_df['total_cost'].std()

print("Score Statistics:")
print(f"Mean: {mean_score}")
print(f"Median: {median_score}")
print(f"Min: {min_score}")
print(f"Max: {max_score}")
print(f"Standard Deviation: {std_dev}")

# Define logarithmic bins for the score range
log_bins = np.logspace(0, np.log10(merged_df['total_cost'].max() + 1), 50)  # 50 bins from 1 to max score + 1

# Plot histogram with logarithmic bins
plt.figure(figsize=(12, 6))
counts, bins, patches = plt.hist(merged_df['total_cost'], bins=log_bins, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
plt.axvline(median_score, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_score:.2f}')

# Add value range labels to each bin
for count, bin_left, bin_right, patch in zip(counts, bins[:-1], bins[1:], patches):
    # Calculate the midpoint for the bin
    bin_midpoint = (bin_left + bin_right) / 2
    # Only add labels to bins with non-zero counts for clarity
    if count > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, count + 5, f'{bin_midpoint:.2f}',
                 ha='center', va='bottom', fontsize=8, rotation=90)

plt.xscale('log')  # Set x-axis to logarithmic scale for better clarity in high range
plt.title('Logarithmic Distribution of Total Cost Scores for 5000 Tests')
plt.xlabel('Total Cost Score (Log Scale)')
plt.ylabel('Frequency')
plt.legend()
plt.show()