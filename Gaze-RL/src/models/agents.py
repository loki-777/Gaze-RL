import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.networks import CNN, GazeAttnCNN
from torchvision.models import resnet18

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18

# Channel-based gaze extractor (uses your new CNN)
class ChannelGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = CNN(use_gaze=True)
    
    def forward(self, observations):
        # Process observations
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        
        # Normalize RGB (first 3 channels)
        rgb = observations[:, :3].float() / 255.0
        
        # Extract gaze channel if available (4th channel)
        if observations.shape[1] > 3:
            gaze = observations[:, 3:4].float() / 255.0
        else:
            # If no gaze channel, create empty tensor
            gaze = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), 
                               device=rgb.device)
        
        # Forward pass through CNN
        return self.cnn(rgb, gaze)

# Bottleneck-based gaze extractor (uses your GazeAttnCNN)
# Bottleneck-based gaze extractor (uses your GazeAttnCNN)
# Bottleneck-based gaze extractor (uses your GazeAttnCNN)
class BottleneckGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.attn_cnn = GazeAttnCNN(use_gaze=True)
    
    def forward(self, observations):
        # Process observations
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        
        # Normalize RGB
        rgb = observations[:, :3].float() / 255.0
        
        # Extract gaze channel if available
        if observations.shape[1] > 3:
            gaze = observations[:, 3:4].float() / 255.0
        else:
            gaze = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), 
                              device=rgb.device)
        
        # Forward pass through attention CNN
        return self.attn_cnn(rgb, gaze)

# Weighted-based gaze extractor
class WeightedGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Main CNN backbone
        self.backbone = resnet18(pretrained=True)
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, features_dim)
        
        # Gaze processor
        self.gaze_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),  # 3 channels to match RGB
            nn.Sigmoid()  # Outputs weights between 0 and 1
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(features_dim)
    
    def forward(self, observations):
        # Process observations
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        
        # Normalize RGB
        rgb = observations[:, :3].float() / 255.0  # First 3 channels are RGB
        
        # Extract gaze channel if available (4-channel input)
        if observations.shape[1] > 3:
            gaze = observations[:, 3:4].float() / 255.0  # 4th channel is gaze
        else:
            # If no gaze channel, create empty tensor
            gaze = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), 
                              device=rgb.device)
        
        # Process gaze to create attention weights
        attention_weights = self.gaze_processor(gaze)
        
        # Apply attention weights to RGB input (element-wise multiplication)
        modulated_input = rgb * attention_weights
        
        # Pass modulated input through backbone
        features = self.backbone(modulated_input)
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features

# Modified GazePPO class to select the right extractor
class GazePPO(PPO):
    def __init__(self, env=None, config=None, policy="CnnPolicy", **kwargs):
        # Extract configuration
        self.config = config or {}
        self.use_gaze = self.config.get("use_gaze", False)
        
        # Get integration method from config or use default
        integration_method = self.config.get("integration_method", "channel")
        
        # Select the appropriate feature extractor based on integration method
        if integration_method == "bottleneck":
            features_extractor_class = BottleneckGazeExtractor
        elif integration_method == "weighted":
            features_extractor_class = WeightedGazeExtractor
        else:  # Default to channel
            features_extractor_class = ChannelGazeExtractor
        
        # Define policy kwargs
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": {}
        })
        
        # Extract PPO hyperparameters from config
        ppo_kwargs = {
            "learning_rate": self.config.get("lr", 3e-4),
            "n_steps": self.config.get("n_steps", 128),
            "batch_size": self.config.get("batch_size", 64),
            "n_epochs": self.config.get("n_epochs", 4),
            "gamma": self.config.get("gamma", 0.99),
            "gae_lambda": self.config.get("gae_lambda", 0.95),
            "clip_range": self.config.get("clip_range", 0.2),
            "ent_coef": self.config.get("ent_coef", 0.01),
            "vf_coef": self.config.get("vf_coef", 0.5),
            "max_grad_norm": self.config.get("max_grad_norm", 0.5),
        }
        
        # Override with any provided kwargs
        ppo_kwargs.update(kwargs)
        
        # Initialize PPO
        super().__init__(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **ppo_kwargs
        )
        
        # Log configuration
        print(f"Initialized GazePPO with use_gaze={self.use_gaze}, integration_method={integration_method}")

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict method to perform additional processing if needed.
        
        Args:
            observation: Environment observation
            state: RNN state (if applicable)
            episode_start: Episode start signals
            deterministic: Whether to sample or use mode
            
        Returns:
            tuple: (actions, states)
        """
        # Call parent's predict method
        return super().predict(
            observation, 
            state=state, 
            episode_start=episode_start, 
            deterministic=deterministic
        )