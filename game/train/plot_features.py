import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

# Path to simulation data directory
data_dir = '../../data'

# Initialize data collectors
columns_of_interest = ["vEgo", "aEgo", "roll", "targetLateralAcceleration"]
data = {col: [] for col in columns_of_interest}

# Read CSV files
for filename in sorted(os.listdir(data_dir))[:5000]:
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        for col in columns_of_interest:
            data[col].extend(df[col].tolist())

# Analyze distributions and thresholds
for column in columns_of_interest:
    print(f"Analyzing feature: {column}")
    values = np.array(data[column]).reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(values)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    thresholds = [(cluster_centers[i] + cluster_centers[i + 1]) / 2 for i in range(len(cluster_centers) - 1)]

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    for center in cluster_centers:
        plt.axvline(center, color='red', linestyle='--', label=f"Cluster Center: {center:.2f}")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Group data based on thresholds
    groups = {i: [] for i in range(len(cluster_centers))}
    for filename in sorted(os.listdir(data_dir))[:5000]:
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            for i, row in df.iterrows():
                value = row[column]
                for group_idx, center in enumerate(cluster_centers):
                    if group_idx == 0 and value <= thresholds[0]:
                        groups[group_idx].append(row)
                    elif group_idx == len(cluster_centers) - 1 and value > thresholds[-1]:
                        groups[group_idx].append(row)
                    elif group_idx < len(thresholds) and thresholds[group_idx - 1] < value <= thresholds[group_idx]:
                        groups[group_idx].append(row)

    # Plot 2 samples (limited to 40 rows) for each group
    for group_idx, samples in groups.items():
        if samples:
            print(f"\nPlotting samples for {column}, Group {group_idx}")
            for sample_idx in range(min(2, len(samples))):  # Limit to 2 samples
                sample_data = pd.DataFrame(samples).iloc[:40]  # Take first 40 rows of the sample
                plt.figure(figsize=(8, 4))
                plt.plot(sample_data.index, sample_data[column], marker='o', label=f"Sample {sample_idx + 1}")
                plt.title(f"Sample Plot - {column}, Group {group_idx}")
                plt.xlabel("Index")
                plt.ylabel(column)
                plt.legend()
                plt.grid()
                plt.show()
