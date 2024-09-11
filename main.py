import datetime
import os
import pickle

import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    extract_features,
    find_nearest_neighbors,
    load_model,
    log,
    reset_model,
    save_model,
    show_image,
)


def train_model(train_folder):
    images = []
    labels = []
    for filename in os.listdir(train_folder):
        img_path = os.path.join(train_folder, filename)
        img = cv2.imread(img_path)
        plt.close()  # Close the previous figure
        show_image(img, filename)  # Use show_image instead of cv2.imshow
        features = extract_features(img)
        label = input(f"Do you like this image {filename}? (l/d): ")
        images.append(features)
        labels.append(1 if label == "l" else 0)
        log(f"Processed {filename} with label {label}")

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(images, labels)
    return model


def predict_images(model, predict_folder):
    new_features = []
    new_labels = []
    for filename in os.listdir(predict_folder):
        img_path = os.path.join(predict_folder, filename)
        img = cv2.imread(img_path)
        plt.close()  # Close the previous figure
        show_image(img, filename)  # Use show_image instead of cv2.imshow
        features = extract_features(img)
        prediction = model.predict([features])[0]
        predicted_label = "like" if prediction == 1 else "dislike"

        # Add these lines to find and print the nearest neighbors
        distances, indices = find_nearest_neighbors(model, features)
        print(f"Nearest neighbors for {filename}: {indices}")
        print(f"Distances: {distances}")

        actual_label = input(
            f"Predicted: {predicted_label}. Do you like this image {filename}? (l/d): "
        )
        actual_label = 1 if actual_label == "l" else 0
        new_features.append(features)
        new_labels.append(actual_label)
        log(f"Predicted {predicted_label} for {filename}, actual was {actual_label}")

    model.fit(new_features, new_labels)


if __name__ == "__main__":
    train_folder = "data/train"
    predict_folder = "data/predict"

    model = load_model()  # Load the model and print the last trained time
    if model is None:  # If no model was loaded, train a new one
        model = train_model(train_folder)
    else:
        reset = input("Do you want to reset the model? (y/n): ")
        if reset.lower() == "y":
            model = train_model(train_folder)

    predict_images(model, predict_folder)

    save_model(model)  # Save the model and the current time
