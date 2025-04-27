import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CustomCNN(nn.Module):
    """
    Custom CNN for feature extraction with optional gaze input channel.
    Modified ResNet18 to accept 4 channels (RGB + gaze) as input.
    Compatible with Gymnasium.
    """
    def __init__(self, use_gaze=False):
        super().__init__()
        self.use_gaze = use_gaze
        
        # Load pretrained ResNet18
        self.backbone = resnet18(pretrained=True)
        
        # Modify first layer to handle 4 channels if using gaze
        if use_gaze:
            # Get weights from the original first layer
            original_weights = self.backbone.conv1.weight.data
            
            # Create new conv layer with 4 input channels
            self.backbone.conv1 = nn.Conv2d(
                4, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize new layer with weights from original layer
            with torch.no_grad():
                # Copy RGB weights
                self.backbone.conv1.weight.data[:, :3] = original_weights
                
                # Initialize gaze channel weights (copy from red channel)
                self.backbone.conv1.weight.data[:, 3] = original_weights[:, 0]
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 512)
        
        # Add layer normalization
        self.ln = nn.LayerNorm(512)
        
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is 3 (RGB) or 4 (RGB+gaze)
            
        Returns:
            torch.Tensor: Feature tensor of shape (B, 512)
        """
        # Handle channel order - SB3 expects (B, H, W, C) but PyTorch needs (B, C, H, W)
        if len(x.shape) == 4 and x.shape[1] != 3 and x.shape[1] != 4:
            # Format is likely (B, H, W, C), convert to (B, C, H, W)
            if x.shape[3] == 3 or x.shape[3] == 4:
                x = x.permute(0, 3, 1, 2)
        
        # Forward pass through backbone
        features = self.backbone(x)
        
        # Apply layer normalization
        features = self.ln(features)
        
        return features


class GazeAttentionModule(nn.Module):
    """
    Optional attention module that can be used to incorporate gaze information
    into feature extraction process.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # Convolutional layers for processing gaze heatmap
        self.gaze_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Attention fusion layer
        self.fusion = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)
        
    def forward(self, features, gaze):
        """
        Apply gaze-based attention to features.
        
        Args:
            features: Feature tensor from CNN (B, C, H, W)
            gaze: Gaze heatmap (B, 1, H, W)
            
        Returns:
            torch.Tensor: Attended features (B, C, H, W)
        """
        # Process gaze heatmap
        processed_gaze = self.gaze_conv(gaze)
        
        # Resize gaze to match feature map dimensions
        if processed_gaze.shape[2:] != features.shape[2:]:
            processed_gaze = F.interpolate(
                processed_gaze, 
                size=features.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply attention mechanism
        attention = torch.sigmoid(processed_gaze)
        
        # Concatenate features with attention map
        combined = torch.cat([features, attention], dim=1)
        
        # Fuse attention with features
        attended_features = self.fusion(combined)
        
        return attended_features