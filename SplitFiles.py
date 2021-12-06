# Author: Omkar
# Create separate test set that will not be used during training or tuning

import os
import random
import shutil

source = 'images_sports_2021'
dest_train = 'Train Set'
dest_test = 'Test Set'
files = os.listdir(source)
test_files_to_move = int(float(len(files)) * 0.1)    #move 10 % files


# Separate Test Data
if not os.path.exists(dest_test):
    os.makedirs(dest_test)
print("Creating test set")
for file_name in random.sample(files, test_files_to_move):
    shutil.move(os.path.join(source, file_name), dest_test)


files = os.listdir(source)
# Separate remaining files as  Train Data
if not os.path.exists(dest_train):
    os.makedirs(dest_train)
print("Creating train set")
for file_name in files:
    shutil.move(os.path.join(source, file_name), dest_train)

