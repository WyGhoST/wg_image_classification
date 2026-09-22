import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            in_features=16 * 7 * 7,
            out_features=64
        )

        self.dropout = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(
            in_features=64,
            out_features=10
        )

    def forward(self, x):
        # [B, 1, 28, 28]
        x = self.conv1(x)
        x = torch.relu(x)

        # [B, 8, 28, 28] -> [B, 8, 14, 14]
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)

        # [B, 16, 14, 14] -> [B, 16, 7, 7]
        x = self.pool(x)

        # [B, 16, 7, 7] -> [B, 784]
        x = torch.flatten(x, start_dim=1)

        # [B, 784] -> [B, 64]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # [B, 64] -> [B, 10]
        logits = self.fc2(x)

        return logits