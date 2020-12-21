# imports
import pandas as pd

# static declarations
dataset_images_csv = "data/images.csv"
dataset_images = "data/images"

# read csv
profiles = pd.read_csv(dataset_images_csv)
print("------------------------------------Original Profiles------------------------------------")
print(profiles)
