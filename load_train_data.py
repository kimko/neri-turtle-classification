'''
To identify photos that show a turtle carapace, we can use a machine learning technique called convolutional neural networks (CNNs).

Here's an overview of the steps we'll need to follow:

    Load the training data and prepare it for use with a CNN.
    Define a CNN architecture and train it using the prepared training data.
    Load the images from the turtle photo folders and preprocess them for use with the trained CNN.
    Use the trained CNN to classify the images and identify the ones that show a turtle carapace.
'''
import os
import cv2
import numpy as np

def load_train_data(data_dir):
    carapace_dir = os.path.join(data_dir, "carapace")
    non_carapace_dir = os.path.join(data_dir, "non-carapace")

    # Load carapace images
    carapace_images = []
    for filename in os.listdir(carapace_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(carapace_dir, filename)
            # print(filepath)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                carapace_images.append(img)

    # Load non-carapace images
    non_carapace_images = []
    for filename in os.listdir(non_carapace_dir):
        if filename.endswith(".jpg"):
            # print(filepath)
            filepath = os.path.join(non_carapace_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                non_carapace_images.append(img)

    return carapace_images, non_carapace_images