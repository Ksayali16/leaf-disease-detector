import os
from sklearn.model_selection import train_test_split
import shutil

# Define the root directory containing all the data directories
root_dir = "C:\\Users\\Admin\\Documents\\leaf_diseas_detector\\dataset\\tomato_leaves"
# Define the directories containing the data
data_dirs = ["C:\\Users\\Admin\\Documents\\leaf_diseas_detector\\dataset\\tomato_leaves\\Tomato_Septoria_leaf_spot"]
# Define the training, testing, and validation set ratios
train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

# Create lists to store the file paths
train_files = []
test_files = []
val_files =[]

# Iterate through each data directory
for data_dir in data_dirs:
    # Get the file paths in the current directory
    file_paths = [os.path.join(root_dir, data_dir, file) for file in os.listdir(os.path.join(root_dir, data_dir))]

    # Split the file paths into training, testing, and validation sets
    train, test_val = train_test_split(file_paths, test_size=test_ratio + val_ratio, random_state=42)
    test, val = train_test_split(test_val, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

    # Add the file paths to the corresponding lists
    train_files.extend(train)
    test_files.extend(test)
    val_files.extend(val)

# Create the training, testing, and validation directories
train_dir = os.path.join(root_dir, 'train_Septoria_leaf_spot')
test_dir = os.path.join(root_dir, 'test_Septoria_leaf_spot')
val_dir = os.path.join(root_dir, 'val_Septoria_leaf_spot')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Move the files to the corresponding directories
for file in train_files:
    shutil.move(file, train_dir)

for file in test_files:
    shutil.move(file, test_dir)

for file in val_files:
    shutil.move(file, val_dir)
