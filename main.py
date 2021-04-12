# imports
import joblib
import numpy as np
import pandas as pd
from glob import glob
import face_recognition
from pathlib import Path as p
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


print("------------------------------------Getting Encoding data from Images------------------------------------")

# collect all face encodings
face_encodings_all = []


def get_face_encoding(image_path):
    # load image in face-recognition
    input_image = face_recognition.load_image_file(image_path, mode='RGB')
    # get face data encodings extracted from image from facenet's pretrained data
    try:
        face_encoding = face_recognition.face_encodings(
            input_image, num_jitters=20, model='large')[0]
        return face_encoding.tolist()
    except IndexError:
        print("Face not found in {}".format(image_path))
        return np.zeros(128).tolist()


# get face encodings for each image in dataframe df
for images in data_df.path:
    face_enc = get_face_encoding(images)
    face_encodings_all.append(face_enc)

print("------------------------------------Training------------------------------------")

# load training data matrix as numpy array
X = np.array(face_encodings_all)
y_height = data_df["Height (m)"].values
y_weight = data_df["Weight (kg)"].values

X_train, X_test, y_height_train, y_height_test, y_weight_train, y_weight_test = \
    train_test_split(X, y_height, y_weight, random_state=1)

# Fit face-encoding data with height as a linear model
model_height = KernelRidge()
model_height = model_height.fit(X_train, np.log(y_height_train))

# Fit face-encoding data with weight as a linear model
model_weight = KernelRidge()
model_weight = model_weight.fit(X_train, np.log(y_weight_train))

print("------------------------------------DONE------------------------------------")

print("------------------------------------Output Model.pkl------------------------------------")


def measure_perf(model, X_test, y_test):
    # Make predictions using the testing set
    y_pred = model.predict(X_test)
    y_true = np.log(y_test)

    errors = abs(y_pred - y_true)
    mean_absolute_percentage_error = np.mean(errors / y_true)
    accuracy = (1 - mean_absolute_percentage_error) * 100

    return {"Average Error": mean_absolute_percentage_error * 100, "Accuracy": accuracy}


# Show model score
print("Height Perf => {}".format(
    measure_perf(model_height, X_test, y_height_test))
)
print("Weight Perf => {}".format(
    measure_perf(model_weight, X_test, y_weight_test))
)


# save model dump into binaries
joblib.dump(model_height, "out/height_predictor.model")
joblib.dump(model_weight, "out/weight_predictor.model")
