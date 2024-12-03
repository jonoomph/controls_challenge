import json
import os

def parse_and_save_levels(input_file, output_file):
    levels_data = {}

    with open(input_file, 'r') as file:
        for line in file:
            # Parse the level number (before the colon)
            level_str, rest = line.split(':', 1)
            level = level_str.strip()

            # Parse the first score (between `:` and `cost`)
            score_str = rest.split('cost')[0].split(':')[-1].strip()
            score = float(score_str)

            # Save to dictionary
            levels_data[level] = score

    # Sort dictionary by score in descending order
    sorted_levels = dict(sorted(levels_data.items(), key=lambda item: item[1], reverse=True))

    # Save to JSON
    with open(output_file, 'w') as json_file:
        json.dump(sorted_levels, json_file, indent=4)
        print(f"Data saved to {output_file}")

    # Save sorted level keys as a list to a secondary JSON
    levels_list = [int(file) for file in list(sorted_levels.keys())]
    with open('../data/levels.json', 'w') as json_file:
        json.dump(levels_list, json_file, indent=4)

    # Remove files in ../data that are not in levels_list
    remove_unused_levels(levels_list, "../data")

def remove_unused_levels(levels_list, data_dir):
    # Convert levels list to a set for faster lookups
    levels_set = set(levels_list)

    # Iterate through files in the data directory
    for file_name in os.listdir(data_dir):
        # Check if the file is an .npy file and extract the level number
        if file_name.endswith('.npy'):
            level_number = int(file_name.split('.')[0])
            if level_number not in levels_set:
                # If the level is not in the list, remove the file
                file_path = os.path.join(data_dir, file_name)
                os.remove(file_path)
                print(f"Removed unused level file: {file_path}")

# Example usage
input_file = 'simulator-output.txt'  # Replace with your input file name
output_file = '../data/high_scores.json'
parse_and_save_levels(input_file, output_file)
