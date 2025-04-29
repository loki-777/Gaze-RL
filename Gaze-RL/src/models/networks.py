import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ChannelCNN(nn.Module):
    def __init__(self, use_gaze=False, feature_dim=512):
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
        self.backbone.fc = nn.Linear(num_features, feature_dim)

        self.ln = nn.LayerNorm(feature_dim)
        
    def forward(self, x, gaze_heatmap):
        if gaze_heatmap:
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
        
        # Cross-attention fusion - adjust for 2D feature maps
        self.query_proj = nn.Conv2d(512, 512, kernel_size=1)
        self.key_proj = nn.Conv2d(512, 512, kernel_size=1)
        self.value_proj = nn.Conv2d(512, 512, kernel_size=1)
        
        self.attention_scale = 512 ** -0.5
        self.num_heads = num_heads
        
        # Output projection
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(self, x, gaze_heatmap):
        # Extract RGB features
        rgb_feats = self.rgb_backbone(x)  # (B, 512, H, W)
        
        if self.use_gaze and gaze_heatmap is not None:
            # Process gaze heatmap
            gaze_feats = self.gaze_encoder(gaze_heatmap)  # (B, 512, 1, 1)
            
            B, C, H, W = rgb_feats.shape
            
            # Expand gaze features to match RGB spatial dimensions
            gaze_feats = gaze_feats.expand(B, C, H, W)
            
            # Project to query, key, value
            q = self.query_proj(gaze_feats).view(B, self.num_heads, C // self.num_heads, H * W)
            k = self.key_proj(rgb_feats).view(B, self.num_heads, C // self.num_heads, H * W)
            v = self.value_proj(rgb_feats).view(B, self.num_heads, C // self.num_heads, H * W)
            
            # Transpose for attention
            q = q.transpose(2, 3)  # (B, num_heads, H*W, C//num_heads)
            k = k.transpose(2, 3)  # (B, num_heads, H*W, C//num_heads)
            v = v.transpose(2, 3)  # (B, num_heads, H*W, C//num_heads)
            
            # Calculate attention scores
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # (B, num_heads, H*W, C//num_heads)
            out = out.transpose(2, 3).contiguous().view(B, C, H, W)
            
            # Add residual connection
            fused = out + rgb_feats
        else:
            fused = rgb_feats
        
        # Global pooling and final projection
        return self.fc(fused)

class WeightedCNN:
    def __init__(self, use_gaze=False, feature_dim=512):
        super().__init__()
        self.use_gaze = use_gaze
        # Main CNN backbone
        self.backbone = resnet18(pretrained=True)
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, feature_dim)
        
        # Gaze processor
        self.gaze_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),  # 3 channels to match RGB
            nn.Sigmoid()  # Outputs weights between 0 and 1
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, gaze_heatmap=None):
        if gaze_heatmap is not None:
            attention_weights = self.gaze_processor(gaze_heatmap)
            modulated_input = attention_weights * x
            features = self.backbone(modulated_input)
        else:
            features = self.backbone(x)
        features = self.layer_norm(features)
        
        return features