import os
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms

# Define transformations to be applied to input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the trained model
model = torch.load("turtle_classifier.pt")

# Define a function to classify images
def classify_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = transform(img).unsqueeze(0)
    output = model(img)
    prediction = torch.max(output, dim=1)[1].item()
    if prediction == 1:
        return True
    else:
        return False

# Set the path to the directory containing the images to be classified
image_dir = "/path/to/images"

# Classify each image in the directory and print the result
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        filepath = os.path.join(image_dir, filename)
        if classify_image(filepath):
            print("{} has a carapace".format(filepath))
        else:
            print("{} does not have a carapace".format(filepath))
