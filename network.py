import torch
from torch import nn

class ConvNet (nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm3d(num_features=8),
            nn.ReLU(),

            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm3d(num_features=16),
            nn.ReLU(),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=55, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)

        return x


