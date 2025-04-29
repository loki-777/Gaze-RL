import torch.nn as nn
from torchvision.models import resnet18
from segmentation_models_pytorch import Unet

# class UNET(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
        
#         self.unet = Unet(
#             encoder_name="resnet34",
#             encoder_weights="imagenet" if pretrained else None,
#             in_channels=3,
#             classes=1,
#             activation='sigmoid'
#         )
        
#         self.final_conv = nn.Conv2d(1, 1, kernel_size=1)

#     def forward(self, x):
#         x = self.unet(x)
#         x = self.final_conv(x)
#         return x


class RESNET(nn.Module):
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