## TODO

import torch.nn as nn
from torchvision.models import resnet18

class CustomCNN(nn.Module):
    def __init__(self, use_gaze=False):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        if use_gaze:
            self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7)  # RGB + Gaze