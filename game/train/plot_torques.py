import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import linregress

# Path to the directory containing your CSV files
data_dir = '../../data'

# List to store relationship summary for each drive
drives_relationship_summary = []

# Iterate over all the CSV files in the directory
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)

        # Read the first 80 rows of data with relevant columns
        df = pd.read_csv(file_path).head(80)

        # Extract steerCommand and targetLateralAcceleration
        steer_command = df['steerCommand']
        target_lat_accel = df['targetLateralAcceleration']

        # Check for zero variance in steerCommand or targetLateralAcceleration
        if steer_command.nunique() == 1 or target_lat_accel.nunique() == 1:
            print(f"Skipping file {filename} due to zero variance in steerCommand or targetLateralAcceleration.")
            continue

        # Calculate the correlation and regression slope for initial analysis
        correlation = np.corrcoef(steer_command, target_lat_accel)[0, 1]
        slope, intercept, r_value, p_value, std_err = linregress(steer_command, target_lat_accel)

        # Determine relationship type (linear vs non-linear) based on R-squared value
        relationship_type = "linear" if r_value ** 2 > 0.8 else "non-linear"

        # Store the summary for clustering analysis
        relationship_summary = {
            'file': filename,
            'correlation': correlation,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'relationship_type': relationship_type
        }
        drives_relationship_summary.append(relationship_summary)

# Convert to DataFrame for further analysis
relationship_df = pd.DataFrame(drives_relationship_summary)

# Normalize for clustering
scaler = StandardScaler()
relationship_df_scaled = scaler.fit_transform(relationship_df[['correlation', 'slope', 'r_squared']])

# Use k-means clustering to cluster drives based on relationship characteristics
kmeans = KMeans(n_clusters=10, random_state=42)
relationship_df['cluster'] = kmeans.fit_predict(relationship_df_scaled)

# Select a random drive from each cluster to visualize different types of relationships
representative_drives = relationship_df.groupby('cluster').apply(lambda x: x.sample(1, random_state=42)).reset_index(
    drop=True)

# Plot representative relationships for each cluster
plt.figure(figsize=(15, 12))
for i, row in representative_drives.iterrows():
    file_path = os.path.join(data_dir, row['file'])
    df = pd.read_csv(file_path).head(80)
    steer_command = df['steerCommand']
    target_lat_accel = df['targetLateralAcceleration']

    plt.subplot(5, 2, i + 1)
    plt.scatter(steer_command, target_lat_accel, color='blue', alpha=0.6)
    plt.title(f"Cluster {row['cluster']} - {row['relationship_type'].capitalize()}\nR²={row['r_squared']:.2f}")
    plt.xlabel('Steer Command')
    plt.ylabel('Target Lateral Acceleration')

    # If relationship is linear, add trend line
    if row['relationship_type'] == "linear":
        plt.plot(steer_command, row['slope'] * steer_command + row['intercept'], color='red', linewidth=2)

plt.tight_layout()
plt.show()

# -------- PCA Visualization --------

# Apply PCA for a 2D overview of clusters
pca = PCA(n_components=2)
pca_components = pca.fit_transform(relationship_df_scaled)

# Create a new DataFrame for plotting
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = relationship_df['cluster']

# Plot the PCA visualization
plt.figure(figsize=(10, 7))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['cluster'], cmap='viridis', s=20, alpha=0.5)
plt.title('PCA of Torque and Target Lateral Acceleration Relationships (Clusters)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
