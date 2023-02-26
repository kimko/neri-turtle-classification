import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from classifier import TurtleClassifier
from helpers import resize_and_rotate_image

DATA_DIR = "/Volumes/T7/turtles_training_data"

class TurtleDataset(Dataset):
    def __init__(self, subject_images, non_subject_images, transform=None):
        self.images = subject_images + non_subject_images
        self.labels = [1] * len(subject_images) + [0] * len(non_subject_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_images(current_subject, current_subject_list, a, b):
    print(f"Loading {current_subject} training images...")
    subject_dir = os.path.join(DATA_DIR, current_subject)
    for filename in os.listdir(subject_dir):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(subject_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            # img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = resize_and_rotate_image(img, a, b, filename)
                current_subject_list.append(img)


def train_model(a=512, b=384, subject="carapace"):
    # Load training images
    subject_images = []
    load_images(subject, subject_images, a, b)
    print(len(subject_images))

    non_subject_images = []
    load_images(f"non-{subject}", non_subject_images, a, b)
    print(len(non_subject_images))

    # Combine images and labels
    dataset = TurtleDataset(
        subject_images, non_subject_images, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    # Define CNN architecture
    # in_channels=3 = 3 color channels (RGB)
    # in_channels=1 = 1 color channel (grayscale)
    model = TurtleClassifier(in_channels=3, height=128, width=96)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # # Define learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Train CNN
    num_epochs = 5
    for epoch in range(num_epochs):
        # Train model
        running_loss = 0.0
        model.train()
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                print(f"Training: Epoch {epoch+1}, batch {i+1}: loss {running_loss/20}")
                running_loss = 0.0

    # Adjust learning rate
    # scheduler.step(val_loss/len(val_dataloader))

    print(f"Finished training {subject} classifier")

    # Save the trained model
    model_path = os.path.join(DATA_DIR, f"{subject}_classifier.pt")
    torch.save(model.state_dict(), model_path)


train_model(a=512, b=384, subject="plastron")
