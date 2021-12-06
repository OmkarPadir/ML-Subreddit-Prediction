# Author: Omkar
# Rename Images to avoid overwriting due to same names

import os

source = 'images19'

files = os.listdir(source)


for file in files:
    old_file = os.path.join(source, file)
    fileNameParts = file.split("-")


    new_file = os.path.join(source,fileNameParts[0]+"_19_"+fileNameParts[1] )
    os.rename(old_file, new_file)



