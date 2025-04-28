import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.networks import *

class SimpleGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, config, observation_space, features_dim=512):
        # Initialize the parent class first
        super().__init__(observation_space, features_dim=512)

        if config["model"]["network"] == "CNN":
            self.network = CNN(use_gaze=config["model"]["use_gaze"])
        if config["model"]["network"] == "Attn":
            self.network = GazeAttnCNN(use_gaze=config["model"]["use_gaze"])
        
        # Store gaze heatmap
        self.gaze_heatmap = None
    
    def forward(self, observations):
        # Process observations
        rgb = observations.float() / 255.0
        
        # Add gaze channel if available
        if self.gaze_heatmap is not None:
            gaze = self.gaze_heatmap
            if len(gaze.shape) == 3:  # (B, H, W)
                gaze = gaze.unsqueeze(1)  # (B, 1, H, W)
        else:
            gaze = None
        
        # Pass through CNN
        return self.network(rgb, gaze)
    
    def set_gaze_heatmap(self, heatmap):
        self.gaze_heatmap = heatmap


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
        self.use_gaze = self.config.get("use_gaze", False)
        
        # Define policy kwargs
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({
            "features_extractor_class": SimpleGazeExtractor,
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
        print(f"Initialized GazePPO with use_gaze={self.use_gaze}")
        
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