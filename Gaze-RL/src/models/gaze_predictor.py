## TODO

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class GazeDataset(Dataset):
    """Loads SALICON dataset (RGB images + gaze heatmaps)"""
    def __init__(self, img_dir, gaze_dir, transform=None):
        self.img_dir = img_dir
        self.gaze_dir = gaze_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        gaze_path = os.path.join(self.gaze_dir, self.img_names[idx].replace('.jpg', '.png'))

        # Load and preprocess image (RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # ResNet input size
        img = img / 255.0  # Normalize to [0, 1]

        # Load and preprocess gaze heatmap
        gaze = cv2.imread(gaze_path, cv2.IMREAD_GRAYSCALE)
        gaze = cv2.resize(gaze, (224, 224))
        gaze = gaze.astype(np.float32) / 255.0  # Normalize to [0, 1]

        if self.transform:
            img = self.transform(img)
            gaze = self.transform(gaze)

        return torch.FloatTensor(img).permute(2, 0, 1), torch.FloatTensor(gaze).unsqueeze(0)

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

def train_gaze_predictor(data_dir, epochs=10, batch_size=32):
    """Train the gaze predictor model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    train_dataset = GazeDataset(
        img_dir=os.path.join(data_dir, 'images'),
        gaze_dir=os.path.join(data_dir, 'gaze_heatmaps')
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = GazePredictor().to(device)
    criterion = nn.MSELoss()  # Pixel-wise mean squared error
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        for imgs, gaze_true in train_loader:
            imgs, gaze_true = imgs.to(device), gaze_true.to(device)
            
            optimizer.zero_grad()
            gaze_pred = model(imgs)
            loss = criterion(gaze_pred, gaze_true)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # Save trained model
    torch.save(model.state_dict(), 'gaze_predictor.pth')
    return model

if __name__ == '__main__':
    train_gaze_predictor(data_dir='path/to/salicon')