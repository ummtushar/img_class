import os
import random
import shutil

# Define paths
data_folder = 'symbols'

# Define train-test split ratio
split_ratio = 0.8

# Load all PNG files
all_files = [f for f in os.listdir(data_folder) if f.endswith('.png')]

# Shuffle the data
random.shuffle(all_files)

# Calculate split index
split_index = int(len(all_files) * split_ratio)

# Assign labels for train and test
train_label = 'train_'
test_label = 'test_'

# Assign labels to files
train_files = [(train_label + file) for file in all_files[:split_index]]
test_files = [(test_label + file) for file in all_files[split_index:]]

# Rename and move the files
for file in train_files:
    os.rename(os.path.join(data_folder, file[len(train_label):]), os.path.join(data_folder, file))

for file in test_files:
    os.rename(os.path.join(data_folder, file[len(test_label):]), os.path.join(data_folder, file))

print("Data split complete.")
