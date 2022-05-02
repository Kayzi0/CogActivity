import torch
from torch import nn
from torchsummary import summary

class ConvNet (nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=55)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    summary(ConvNet(), (9, 200, 3))