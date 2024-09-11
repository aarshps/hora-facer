import datetime
import os
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    features = resized.flatten()
    return features


def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def show_image(img, title):
    plt.ion()  # Turn on interactive mode
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.draw()  # Draw the figure without blocking execution
    plt.pause(0.001)  # Pause for a short period to allow the figure to update


def save_model(model, model_path="model.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump((model, datetime.datetime.now()), f)


def load_model(model_path="model.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model, last_trained = pickle.load(f)
        print(f"Model was last trained at {last_trained}")
        return model
    else:
        return None


def reset_model(model_path="model.pkl"):
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Model reset successfully.")


def find_nearest_neighbors(model, img_features, n_neighbors=3):
    distances, indices = model.kneighbors([img_features], n_neighbors)
    return distances, indices


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    features = resized.flatten()
    return features
