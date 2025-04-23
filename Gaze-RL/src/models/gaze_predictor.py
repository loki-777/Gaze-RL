## TODO

import torch
import torch.nn as nn
from torchvision.models import resnet18

# or UNET?

class GazePredictor(nn.Module):
    """CNN to predict gaze heatmaps from RGB images"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        
        # Replace final layer for regression
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 224 * 224),
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1, 224, 224)  # Reshape to heatmap