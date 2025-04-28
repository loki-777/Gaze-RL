import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CNN(nn.Module):
    def __init__(self, use_gaze=False):
        super().__init__()
        self.use_gaze = use_gaze
        self.backbone = resnet18(pretrained=True)
        
        if use_gaze:
            original_weights = self.backbone.conv1.weight.data
            self.backbone.conv1 = nn.Conv2d(
                4, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            with torch.no_grad():
                self.backbone.conv1.weight.data[:, :3] = original_weights
                self.backbone.conv1.weight.data[:, 3] = original_weights[:, 0]
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 512)

        self.ln = nn.LayerNorm(512)
        
    def forward(self, x, gaze_heatmap):
        x = torch.cat([x, gaze_heatmap], dim=1)
        if len(x.shape) == 4 and x.shape[1] != 3 and x.shape[1] != 4:
            if x.shape[3] == 3 or x.shape[3] == 4:
                x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        features = self.ln(features)
        
        return features



class GazeAttnCNN(nn.Module):
    def __init__(self, use_gaze=False, num_heads=8):
        super().__init__()
        self.use_gaze = use_gaze
        
        # RGB backbone (ResNet18 up to layer4)
        self.rgb_backbone = resnet18(pretrained=True)
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        
        # Gaze encoder (lightweight)
        self.gaze_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # Global context
        )
        
        # Cross-attention fusion
        self.fusion = nn.MultiheadAttention(512, num_heads)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512)
        )

        self.norm_q = nn.LayerNorm(512)
        self.norm_kv = nn.LayerNorm(512)
        self.norm_out = nn.LayerNorm(512)

        self.to_q = nn.Linear(512, 512)  # Gaze -> Queries
        self.to_kv = nn.Linear(512, 2 * 512)  # RGB -> Keys/Values

    def forward(self, x, gaze_heatmap):
        rgb_feats = self.rgb_backbone(x)
        
        if self.use_gaze and gaze_heatmap is not None:
            gaze_feats = self.gaze_encoder(gaze_heatmap)
            B, C, H, W = rgb_feats.shape
            gaze_feats = gaze_feats.expand(B, C, H, W)
            
            q = self.norm_q(gaze_feats)
            kv = self.norm_kv(rgb_feats)
            
            q = self.to_q(q)
            k, v = self.to_kv(kv).chunk(2, dim=-1)
            fused = self.fusion(q, k, v)
        else:
            fused = rgb_feats
        
        out = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)
        return self.projection(out)
