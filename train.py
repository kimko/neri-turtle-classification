import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class TurtleDataset(Dataset):
    def __init__(self, carapace_images, non_carapace_images, transform=None):
        self.images = carapace_images + non_carapace_images
        self.labels = [1]*len(carapace_images) + [0]*len(non_carapace_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model():
    data_dir = '/Volumes/T7/turtles_training_data'
    carapace_dir = os.path.join(data_dir, "carapace")
    non_carapace_dir = os.path.join(data_dir, "non-carapace")

    # Load carapace images
    carapace_images = []
    for filename in os.listdir(carapace_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(carapace_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                carapace_images.append(img)

    # Load non-carapace images
    non_carapace_images = []
    for filename in os.listdir(non_carapace_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(non_carapace_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                non_carapace_images.append(img)

    # Combine images and labels
    dataset = TurtleDataset(carapace_images, non_carapace_images, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    # Define CNN architecture
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 25 * 25, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1)
    )

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Train CNN
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                print(f'Epoch {epoch+1}, batch {i+1}: loss {running_loss/20}')
                running_loss = 0.0

    print('Finished training')

train_model()
