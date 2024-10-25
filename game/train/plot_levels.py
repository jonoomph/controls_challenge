import pandas as pd
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to the directory containing your CSV files
data_dir = '../../data'

# List to store summary statistics for each drive
drives_summary = []

# Iterate over all the CSV files in the directory
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)

        # Read the first 600 rows of data
        df = pd.read_csv(file_path).head(600)

        # Compute statistics for all 600 rows for non-steerCommand columns
        drive_summary = {
            'file': filename,
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

# Normalize the summary statistics for clustering
scaler = StandardScaler()
summary_df_scaled = scaler.fit_transform(summary_df[['vEgo_mean', 'vEgo_std', 'aEgo_mean', 'aEgo_std',
                                                     'targetLateralAcceleration_mean', 'targetLateralAcceleration_std',
                                                     'steerCommand_mean', 'steerCommand_std']])

# Use k-means clustering to cluster the drives into groups
kmeans = KMeans(n_clusters=100, random_state=44)
summary_df['cluster'] = kmeans.fit_predict(summary_df_scaled)

# Select one random drive from each cluster to get 100 representative drives
representative_drives = summary_df.groupby('cluster').apply(lambda x: x.sample(1, random_state=44)).reset_index(drop=True)

# Display the selected representative drives
print(representative_drives[['file', 'vEgo_mean', 'aEgo_mean', 'steerCommand_mean']])

# Extract the file numbers without the '.csv' extension
file_numbers = sorted(representative_drives['file'].str.replace('.csv', '', regex=False).astype(int).tolist())

# Save the list of file numbers to a JSON file
with open('../data/levels.json', 'w') as json_file:
    json.dump(file_numbers, json_file)

print(f'Successfully saved {len(file_numbers)} file numbers to levels.json')

# ---------- Visualization Section ------------------

# Dimensionality reduction using PCA (reduce to 2 components)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(summary_df_scaled)

# Create a new DataFrame for plotting
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = summary_df['cluster']

# Add a flag for the representative picks
pca_df['is_representative'] = 0
pca_df.loc[representative_drives.index, 'is_representative'] = 1

# Plot the full dataset and highlight the 100 representative picks
plt.figure(figsize=(10, 7))

# Plot all points in the dataset
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['cluster'], cmap='viridis', s=10, alpha=0.5, label="Data Points")

# Highlight the representative picks
plt.scatter(pca_df[pca_df['is_representative'] == 1]['PCA1'],
            pca_df[pca_df['is_representative'] == 1]['PCA2'],
            color='red', s=50, label="Representative Picks", edgecolor='black')

plt.title('PCA of Drive Data (Clusters and Representative Picks)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
