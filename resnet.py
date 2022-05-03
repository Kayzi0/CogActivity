import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F

class ResNetBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.left = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2),
            nn.BatchNorm1d(out_channels)
        )

        self.right = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=17, stride = 2, padding = 8),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=17, padding = 8),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        x = left + right
        x = F.relu(x)
        return x
          

class ResNet (nn.Module):
    def __init__(self):
        super().__init__()

        self.res1 = ResNetBlock(24,32)
        self.res2 = ResNetBlock(32,64)
        self.res3 = ResNetBlock(64,128)
        self.res4 = ResNetBlock(128,128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,55)
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    summary(ResNet(), (24, 200))