import os
import cv2
import torch
from torch.utils.data import Dataset

class TurtleDataset(Dataset):
    def __init__(self, data_dir):
        self.carapace_dir = os.path.join(data_dir, "carapace")
        self.non_carapace_dir = os.path.join(data_dir, "non-carapace")

        # Load carapace images
        carapace_images = []
        for filename in os.listdir(self.carapace_dir):
            if filename.endswith(".jpg"):
                filepath = os.path.join(self.carapace_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    carapace_images.append(img)

        # Load non-carapace images
        non_carapace_images = []
        for filename in os.listdir(self.non_carapace_dir):
            if filename.endswith(".jpg"):
                filepath = os.path.join(self.non_carapace_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    non_carapace_images.append(img)

        # Create labels for carapace and non-carapace images
        self.carapace_labels = torch.ones(len(carapace_images), dtype=torch.long)
        self.non_carapace_labels = torch.zeros(len(non_carapace_images), dtype=torch.long)

        # Combine images and labels
        self.images = carapace_images + non_carapace_images
        self.labels = torch.cat([self.carapace_labels, self.non_carapace_labels])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        return img_tensor, label
