import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

# v1 - ResidualBlocks 32->64->128->256->512->512
"""class PlateDetectorGrid(nn.Module):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=2)
        )

        self.detector_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector_head(x)

        conf = torch.sigmoid(x[:, 0:1, :, :])
        bboxes = torch.sigmoid(x[:, 1:5, :, :])

        return torch.cat([conf, bboxes], dim=1).permute(0, 2, 3, 1)"""

# v2 - DSBlocks 32->64->128->256->256
class PlateDetectorGrid(nn.Module):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=2),
            ResidualBlock(256, 256, stride=2)
        )

        self.detector_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector_head(x)

        conf = torch.sigmoid(x[:, 0:1, :, :])
        bboxes = torch.sigmoid(x[:, 1:5, :, :])

        return torch.cat([conf, bboxes], dim=1).permute(0, 2, 3, 1)

