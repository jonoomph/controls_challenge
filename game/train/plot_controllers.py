import pandas as pd
import matplotlib.pyplot as plt

# Load the single CSV file into a DataFrame
data = pd.read_csv("../../eval-pids-5000.csv")

# Pivot the data to compare controllers
pivoted_data = data.pivot(index='file', columns='controller', values=['jerk_cost', 'lataccel_cost', 'total_cost'])

# Flatten the MultiIndex columns for easier access
pivoted_data.columns = ['_'.join(col) for col in pivoted_data.columns]
pivoted_data.reset_index(inplace=True)

# Filter files where `pid_model` performs worse than all other controllers
metrics = ['jerk_cost', 'lataccel_cost', 'total_cost']
sorted_worse_than_all = []

for metric in metrics:
    pid_column = f"{metric}_pid_model"
    other_columns = [col for col in pivoted_data.columns if metric in col and 'pid_model' not in col]

    # Calculate the difference between pid_model and the max of other controllers
    pivoted_data['worst_diff'] = pivoted_data[pid_column] - pivoted_data[other_columns].max(axis=1)

    # Filter files where pid_model performs worse than all others
    worse_files = pivoted_data[pivoted_data[pid_column] > pivoted_data[other_columns].max(axis=1)]

    # Sort by the worst difference (descending)
    sorted_files = worse_files.sort_values(by='worst_diff', ascending=False)
    sorted_worse_than_all.append((metric, sorted_files[['file', 'worst_diff']]))

# Print or save the sorted results
for metric, sorted_files in sorted_worse_than_all:
    print(f"Files where pid_model performs worse than all others on {metric} (sorted by severity):")
    print(sorted_files.to_string(index=False))
    print()

# Generate comparative plots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for controller in set(data['controller']):
        col = f"{metric}_{controller}"
        plt.plot(pivoted_data['file'], pivoted_data[col], label=controller)
    plt.title(f"Comparison of {metric} across controllers")
    plt.xlabel("Simulation File")
    plt.ylabel(metric.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
