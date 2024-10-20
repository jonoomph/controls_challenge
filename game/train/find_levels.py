import pandas as pd
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Path to the directory containing your CSV files
data_dir = '../../data'

# List to store summary statistics for each drive
drives_summary = []

# Iterate over all the CSV files in the directory
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)

        # Read the first 100 rows (or less if the file is smaller)
        df = pd.read_csv(file_path).head(100)

        # Extract key summary statistics for the drive
        drive_summary = {
            'file': filename,
            'vEgo_mean': df['vEgo'].mean(),
            'vEgo_std': df['vEgo'].std(),
            'aEgo_mean': df['aEgo'].mean(),
            'aEgo_std': df['aEgo'].std(),
            'targetLateralAcceleration_mean': df['targetLateralAcceleration'].mean(),
            'targetLateralAcceleration_std': df['targetLateralAcceleration'].std(),
            'steerCommand_mean': df['steerCommand'].mean(),
            'steerCommand_std': df['steerCommand'].std(),
        }

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
kmeans = KMeans(n_clusters=100, random_state=42)
summary_df['cluster'] = kmeans.fit_predict(summary_df_scaled)

# Select one random drive from each cluster to get 100 representative drives
representative_drives = summary_df.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)

# Display the selected representative drives
print(representative_drives[['file', 'vEgo_mean', 'aEgo_mean', 'steerCommand_mean']])


# Extract the file numbers without the '.csv' extension
file_numbers = sorted(representative_drives['file'].str.replace('.csv', '', regex=False).astype(int).tolist())

# Save the list of file numbers to a JSON file
with open('../data/levels.json', 'w') as json_file:
    json.dump(file_numbers, json_file)

print(f'Successfully saved {len(file_numbers)} file numbers to levels.json')