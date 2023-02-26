import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from classifier import TurtleClassifier
from PIL import Image
import shutil
import csv
from helpers import resize_and_rotate_image


def classify(
    root_directory, classify_directory, subject, rename_matches=False, threshold=0.8
):
    # Define the transforms to be applied to each image
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the model from the saved state_dict
    model_state_dict = torch.load(
        f"/Volumes/T7/turtles_training_data/{subject}_classifier.pt"
    )
    model = TurtleClassifier(in_channels=3, height=128, width=96)
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
    model.eval()

    identified_subjects = []
    possible_subjects = []
    unique_subjects = set()

    print("Threshold:", threshold)
    # Loop through all images in the directory and classify them
    for subdir, dirs, files in os.walk(root_directory):
        for filename in files:
            if (
                filename.lower().endswith(".jpg")
                and not filename.startswith("._")
                and f"_CARAPACE_" not in filename
            ):
                filepath = os.path.join(subdir, filename)

                # Load the image and convert it to a PIL Image object
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                img = resize_and_rotate_image(img, 512, 384)
                img_pil = Image.fromarray(np.uint8(img))

                # Apply the transformation
                img_tensor = transform(img_pil).unsqueeze(0)

                # Classify the image
                with torch.no_grad():
                    output = model(img_tensor)
                    # pred = torch.argmax(output).item()
                    prob = torch.softmax(output, dim=1)
                    pred_prob, pred_label = torch.max(prob, dim=1)
                    pred_prob = pred_prob.item()
                    pred_label = pred_label.item()

                # Print the filename and classification result
                if pred_label == 1 and pred_prob > threshold:
                    print(f"YES {subdir} {pred_prob:.4f} {filename}")
                    identified_subjects.append((pred_prob, filepath))
                    shutil.copy(filepath, classify_directory + f"/identified_{subject}")

                    filename_parts = filename.split("_")
                    new_filename = f"{filename_parts[0]}_{subject.upper()}_{'_'.join(filename_parts[1:])}"
                    try:
                        unique_subjects.add(int(filename_parts[0]))
                    except:
                        print("BAD filename")
                    if rename_matches == True:
                        if filename_parts[1] != subject.upper():
                            shutil.move(filepath, os.path.join(subdir, new_filename))
                elif pred_label == 1 and pred_prob <= threshold:
                    shutil.copy(filepath, classify_directory + f"/possible_{subject}")
                    possible_subjects.append((pred_prob, filepath))
                    print(f"MAY {subdir} {pred_prob:.4f} {filename}")
                else:
                    print(f"NOP {subdir} {pred_prob:.4f} {filename}")

    with open(f"identified_{subject}_paths.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Probability",
                "Filepath",
            ]
        )
        writer.writerows(identified_subjects)

    with open(f"possible_{subject}_paths.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Probability",
                "Filepath",
            ]
        )
        writer.writerows(possible_subjects)

    with open(f"unique_{subject}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ID",
            ]
        )
        writer.writerows([[f"{id:04}"] for id in sorted(unique_subjects)])


classify("/Volumes/T7/turtles", "/Volumes/T7/classify", "plastron", False, 0.75)
