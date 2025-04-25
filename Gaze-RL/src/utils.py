import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

def calculate_reward(obs, action, target_object="Microwave", gaze_heatmap=None):
    """Calculate reward for the agent.
    
    Args:
        obs: Dictionary containing observation information
        action: Action taken by the agent
        target_object: Target object to search for
        gaze_heatmap: Optional gaze heatmap to guide exploration
        
    Returns:
        float: Calculated reward
    """
    reward = -0.01  # Step penalty to encourage efficiency
    
    # Reward for finding the target object
    if target_object in obs["visible_objects"]:
        # Calculate reward based on object visibility/position
        visibility_score = obs["object_visibility"].get(target_object, 0)
        reward += 2.0 * visibility_score
    
    # Success reward (if object is found and agent is close)
    if obs["success"]:
        reward += 10.0
    
    # Penalty for repeated actions (if provided in obs)
    if obs.get("repeated_action", False):
        reward -= 0.1
        
    # Gaze-guided reward component
    if gaze_heatmap is not None:
        # Extract agent's current view coordinates
        agent_position = obs.get("agent_position", (0, 0))
        
        # Calculate attention overlap using IoU
        attention_score = calculate_attention_iou(agent_position, gaze_heatmap)
        reward += 0.2 * attention_score
    
    return reward

def calculate_attention_iou(agent_position, gaze_heatmap, view_radius=50):
    """Calculate IoU between agent's view area and gaze heatmap.
    
    Args:
        agent_position: (x, y) position of agent in environment
        gaze_heatmap: 2D numpy array representing gaze heatmap
        view_radius: Radius of agent's view area
        
    Returns:
        float: IoU score
    """
    # Create binary mask for agent's view area
    x, y = agent_position
    h, w = gaze_heatmap.shape
    
    # Create agent view mask
    agent_view = np.zeros_like(gaze_heatmap)
    
    # Ensure coordinates are within bounds
    x = min(max(0, x), w-1)
    y = min(max(0, y), h-1)
    
    # Create circular mask for agent's view
    y_indices, x_indices = np.ogrid[:h, :w]
    distance = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
    agent_view[distance <= view_radius] = 1
    
    # Threshold gaze heatmap
    gaze_binary = (gaze_heatmap > 0.2).astype(np.float32)
    
    # Calculate IoU
    intersection = np.logical_and(agent_view, gaze_binary).sum()
    union = np.logical_or(agent_view, gaze_binary).sum()
    
    if union == 0:
        return 0
    
    return intersection / union

def get_exploration_metrics(episode_info):
    """Calculate exploration metrics from episode information.
    
    Args:
        episode_info: Dictionary containing episode information
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "success_rate": float(episode_info["success"]),
        "episode_length": episode_info["episode_length"],
        "cumulative_reward": episode_info["cumulative_reward"],
    }
    
    # Add object-specific metrics
    if "visible_objects" in episode_info:
        metrics["unique_objects_seen"] = len(set(episode_info["visible_objects"]))
    
    # Add exploration coverage
    if "explored_area" in episode_info:
        metrics["exploration_coverage"] = episode_info["explored_area"]
    
    return metrics

def normalize_image(image):
    """Normalize image pixels to range [0, 1]."""
    if image.max() > 1.0:
        return image / 255.0
    return image

# Dataset class for SALICON
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
        heatmap_path = os.path.join(
            self.heatmap_dir, 
            self.img_names[idx].replace('.jpg', '.png')
        )
        
        # Load and preprocess
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = normalize_image(img)
        
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = normalize_image(heatmap)

        if self.transform:
            img = self.transform(img)
            heatmap = self.transform(heatmap)

        return (
            torch.FloatTensor(img).permute(2, 0, 1),  # (3, 224, 224)
            torch.FloatTensor(heatmap).unsqueeze(0)   # (1, 224, 224)
        )

# Environment wrapper for adding gaze features
class GazeEnvWrapper:
    def __init__(self, env, gaze_predictor=None):
        self.env = env
        self.gaze_predictor = gaze_predictor
    
    def reset(self):
        obs = self.env.reset()
        if self.gaze_predictor:
            gaze_heatmap = self._predict_gaze(obs["rgb"])
            obs["gaze_heatmap"] = gaze_heatmap
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.gaze_predictor:
            gaze_heatmap = self._predict_gaze(obs["rgb"])
            obs["gaze_heatmap"] = gaze_heatmap
            # Augment reward with gaze information
            reward = calculate_reward(obs, action, gaze_heatmap=gaze_heatmap)
        return obs, reward, done, info
    
    def _predict_gaze(self, image):
        """Predict gaze heatmap from RGB image."""
        if image.shape != (224, 224, 3):
            image = cv2.resize(image, (224, 224))
        
        # Normalize and convert to tensor
        image = normalize_image(image)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        # Run prediction
        with torch.no_grad():
            heatmap = self.gaze_predictor(image_tensor).squeeze().numpy()
        
        return heatmap