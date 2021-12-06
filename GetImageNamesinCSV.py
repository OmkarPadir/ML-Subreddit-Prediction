#Author: Omkar
# Code used to get a list of downloaded images
# This list is used for filtering un-necessary records in combined model.

import os
import csv

source = 'images19'

files = os.listdir(source)
#open the file in the write mode
f = open('images19List.csv', 'w')
writer = csv.writer(f)
for row in files:
    print(row)
    f.write(row+"\n")


f.close()