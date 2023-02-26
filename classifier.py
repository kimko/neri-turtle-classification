import torch.nn as nn
import torch.nn.functional as F


class TurtleClassifier(nn.Module):
    def __init__(self, in_channels=3, height=1024, width=768):
        super(TurtleClassifier, self).__init__()

        self.height = height
        self.width = width

        # define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # define linear layers
        self.fc1 = nn.Linear(in_features=32 * height * width, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        # apply convolutional layers and activation functions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the tensor for the linear layers
        output_size = 32 * (self.height) * (self.width)
        x = x.view(-1, output_size)
        # x = x.view(-1, 32 * 25 * 25)

        # apply linear layers and activation functions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
