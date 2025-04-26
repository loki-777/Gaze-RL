# should contain helper functions for rewards, loggers, wrappers
## TODO
import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# def calculate_reward(obs, action, gaze_heatmap=None):
#     reward = -0.01  # Step penalty
#     if gaze_heatmap:
#         reward += 0.1 * iou(agent_attention, gaze_heatmap)
#     return reward
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.squeeze().cpu().numpy()

def compute_reward(success, step_taken, gaze_iou, progress_delta,
                   λ=0.5, α=0.1):
    r_success = 1.0 if success else -0.1
    r_step = -0.01 if step_taken else 0
    r_gaze = λ * gaze_iou
    r_progress = α * progress_delta
    return r_success + r_step + r_gaze + r_progress

def compute_gaze_iou(agent_map, gaze_map):
    agent_map = agent_map / (np.sum(agent_map) + 1e-6)
    gaze_map = gaze_map / (np.sum(gaze_map) + 1e-6)
    intersection = np.minimum(agent_map, gaze_map).sum()
    union = np.maximum(agent_map, gaze_map).sum()
    return intersection / (union + 1e-6)

def compute_metrics(episode_rewards, success_flags, steps_list):
    return {
        "avg_reward": np.mean(episode_rewards),
        "success_rate": np.mean(success_flags),
        "avg_steps": np.mean(steps_list),
    }

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


def visualize_predictions(images, ground_truths, predictions, num_samples=5):
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    axes = axes if num_samples > 1 else [axes]

    for i in range(num_samples):
        # Convert image from [3, H, W] to [H, W, 3] for display
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Squeeze ground truth and prediction to [H, W]
        gt = ground_truths[i].squeeze(0).cpu().numpy()
        pred = predictions[i].squeeze(0).detach().numpy()

        # Original image
        axes[i][0].imshow(img)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")

        # Ground truth
        axes[i][1].imshow(gt, cmap="hot")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        # Prediction
        axes[i][2].imshow(pred, cmap="hot")
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")
    
    plt.tight_layout()
    plt.show()