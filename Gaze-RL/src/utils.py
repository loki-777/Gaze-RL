# should contain helper functions for rewards, loggers, wrappers
## TODO
import os
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# def calculate_reward(obs, action, gaze_heatmap=None):
#     reward = -0.01  # Step penalty
#     if gaze_heatmap:
#         reward += 0.1 * iou(agent_attention, gaze_heatmap)
#     return reward

class SALICONDataset(Dataset):
    def __init__(self, img_dir, heatmap_dir, transform=None):
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        heatmap_path = os.path.join(self.heatmap_dir, 
                                  self.img_names[idx].replace('.jpg', '.png'))
        
        # Load and preprocess
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, (224, 224)) / 255.0

        if self.transform:
            img = self.transform(img)
            heatmap = self.transform(heatmap)

        return (
            torch.FloatTensor(img).permute(2, 0, 1),  # (3, 224, 224)
            torch.FloatTensor(heatmap).unsqueeze(0)   # (1, 224, 224)
        )