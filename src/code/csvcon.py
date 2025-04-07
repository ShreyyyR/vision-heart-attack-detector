import os
import pandas as pd

# Directory where your labeled images are stored
base_dir = 'labeled_frames/'

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate through each subdirectory (label)
for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if os.path.isdir(label_dir):
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.jpg'):
                file_paths.append(os.path.join(label_dir, file_name))
                labels.append(label)

# Create a DataFrame
data = pd.DataFrame({'FilePath': file_paths, 'Label': labels})

# Save DataFrame to CSV
data.to_csv('rlabels.csv', index=False)
