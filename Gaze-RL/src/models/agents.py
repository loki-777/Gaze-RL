import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .networks import CustomCNN

class GazeFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor that processes RGB+Gaze input for RL agent.
    Compatible with Gymnasium.
    """
    def __init__(self, observation_space, features_dim=512, use_gaze=False):
        # Initialize parent
        super().__init__(observation_space, features_dim)
        
        # Determine input channels (RGB + optional gaze channel)
        self.use_gaze = use_gaze
        
        # Create CNN feature extractor
        self.cnn = CustomCNN(use_gaze=use_gaze)
        
        # Get feature dimension from CNN
        self.features_dim = features_dim
        
    def forward(self, observations):
        """
        Process observations (images + optional gaze heatmap).
        
        Args:
            observations: For standard mode, this is a tensor of RGB images.
                         If using gaze, we expect the gaze heatmap in the info dict.
            
        Returns:
            torch.Tensor: Features extracted from observations
        """
        # Process RGB images - in Gymnasium/SB3 2.0, observations are typically tensors
        if isinstance(observations, dict):
            # If still using dict observations
            rgb = observations["rgb"].float() / 255.0
        else:
            # Standard case - plain tensor
            rgb = observations.float() / 255.0
        
        # If using gaze, we need to get it from elsewhere
        # In Gymnasium, we would typically put gaze info in the info dict
        # and access it before passing to the model
        if self.use_gaze and hasattr(self, 'gaze_heatmap') and self.gaze_heatmap is not None:
            # Ensure gaze channel has proper shape
            gaze = self.gaze_heatmap
            
            if len(gaze.shape) == 3:  # (B, H, W)
                gaze = gaze.unsqueeze(1)  # Add channel dim: (B, 1, H, W)
            
            # Concatenate RGB and gaze along channel dimension
            x = torch.cat([rgb, gaze], dim=1)  # (B, 4, H, W)
        else:
            x = rgb
        
        # Extract features using CNN
        features = self.cnn(x)
        
        return features
    
    def set_gaze_heatmap(self, gaze_heatmap):
        """Set gaze heatmap for next forward pass."""
        self.gaze_heatmap = gaze_heatmap


class GazePPO(PPO):
    """
    A PPO agent that can use gaze information to guide object search.
    Compatible with Gymnasium.
    """
    def __init__(self, env=None, config=None, policy="CnnPolicy", **kwargs):
        """
        Initialize GazePPO agent.
        
        Args:
            env: Training environment
            config: Configuration dictionary
            policy: Policy class or string
            **kwargs: Additional arguments passed to PPO
        """
        # Extract configuration
        self.config = config or {}
        use_gaze = self.config.get("use_gaze", False)
        
        # Define policy kwargs
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({
            "features_extractor_class": GazeFeatureExtractor,
            "features_extractor_kwargs": {"use_gaze": use_gaze}
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
        
        # Save gaze setting
        self.use_gaze = use_gaze
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict method to handle gaze information.
        
        Args:
            observation: Environment observation
            state: RNN state (if applicable)
            episode_start: Episode start signals
            deterministic: Whether to sample or use mode
            
        Returns:
            tuple: (actions, states)
        """
        # If using gaze, we need to ensure the feature extractor has access to it
        if self.use_gaze and hasattr(self.policy, 'features_extractor'):
            # Check if we have gaze information available
            if isinstance(self.env.buf_infos[0], dict) and "gaze_heatmap" in self.env.buf_infos[0]:
                # Get gaze heatmap from environment info
                gaze_heatmap = self.env.buf_infos[0]["gaze_heatmap"]
                
                # Convert to tensor if needed
                if isinstance(gaze_heatmap, np.ndarray):
                    gaze_heatmap = torch.FloatTensor(gaze_heatmap).to(self.device)
                
                # Pass gaze heatmap to feature extractor
                self.policy.features_extractor.set_gaze_heatmap(gaze_heatmap)
        
        # Call parent's predict method
        return super().predict(
            observation, 
            state=state, 
            episode_start=episode_start, 
            deterministic=deterministic
        )