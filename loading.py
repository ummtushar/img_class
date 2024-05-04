import os
import random
import shutil
from PIL import Image
import numpy as np

# Define paths
fresh_folder = 'Fresh'
spoiled_folder = 'Spoiled'

# Define train-test split ratio
split_ratio = 0.8

# Load all PNG files from Fresh and Spoiled folders
all_files = [(os.path.join(fresh_folder, f), 1) for f in os.listdir(fresh_folder) if f.endswith('.jpg')] + \
            [(os.path.join(spoiled_folder, f), 0) for f in os.listdir(spoiled_folder) if f.endswith('.jpg')]

# Shuffle the data
random.shuffle(all_files)

# Calculate split index
split_index = int(len(all_files) * split_ratio)

# Assign labels to files
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Flatten and label the images
train_data = [(np.array(Image.open(file).convert('L')).flatten(), label) for file, label in train_files]
test_data = [(np.array(Image.open(file).convert('L')).flatten(), label) for file, label in test_files]

print("Data split and image flattening complete.")
print("Train image shape:", train_data[0][0].shape)
print("Test image shape:", test_data[0][0].shape)