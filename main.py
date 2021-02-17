# imports
import pandas as pd
from glob import glob
from pathlib import Path as p

# static declarations
dataset_images_csv = "data/images.csv"
dataset_images = "data/images"

# read csv
profiles = pd.read_csv(dataset_images_csv)
print("------------------------------------Original Profiles------------------------------------")
print(profiles)

print("------------------------------------Loading Images------------------------------------")

slugs = profiles.slug.unique()
# get all images
all_jpgs = []
for slug in slugs:
    profile_images = glob(dataset_images + "/" + slug + "/*")
    count = 0
    # load only jpeg images
    for img in profile_images:
        if ".jpg" in img or ".jpeg" in img or "JPG" in img:
            all_jpgs.append(img)
            count += 1
    print("{:0=3d} images from {}".format(count, slug))

all_jpgs = sorted(all_jpgs)  # all images sorted
print("-------------------")
print("{:0=3d} images in Total".format(len(all_jpgs)))


id_path = [(p(images).stem.split("_")[0], images) for images in all_jpgs]

image_df = pd.DataFrame(id_path, columns=['slug', 'path'])
data_df = image_df.merge(profiles)
print("------------------------------------Data Frame------------------------------------")
print(data_df)
