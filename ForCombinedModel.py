import os
import shutil
import pandas as pd
# from tqdm import tqdm



df1 = pd.read_csv("All Text-Image Cleaned data.csv")
image_source_dir = "Combined"
image_dest_dir = "Merged/"



for i in range(len(df1)):

    img_name = df1.iloc[i, 2]
    img_name += '.jpg'
    img_file = os.path.join(image_source_dir, img_name)


    if os.path.exists(img_file):
        shutil.copy(img_file, image_dest_dir)
    else:
        print(img_name)