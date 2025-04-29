import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.networks import *
from torchvision.models import resnet18

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18

class CommonFeatureExtractor(BaseFeaturesExtractor):
    _config = {
        "network_type": "ChannelCNN",
        "use_gaze": False
    }
    
    @classmethod
    def configure(cls, network_type: str, use_gaze: bool):
        cls._config = {
            "network_type": network_type,
            "use_gaze": use_gaze
        }
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.network = self._create_network()
        
    def _create_network(self):
        network_map = {
            "ChannelCNN": ChannelCNN,
            "GazeAttnCNN": GazeAttnCNN,
            "WeightedCNN": WeightedCNN
        }
        network_class = network_map.get(self._config["network_type"])
        return network_class(use_gaze=self._config["use_gaze"])
    
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
            return self.network(rgb, gaze)


class GazePPO(PPO):
    def __init__(self, env=None, config=None, policy="CnnPolicy", **kwargs):
        # Extract configuration
        self.config = config or {}
        self.use_gaze = self.config.get("use_gaze", False)
        
        CommonFeatureExtractor.configure(
            network_type=config["model"]["features_extractor"],
            use_gaze=config["model"]["use_gaze"]
        )
        
        # Define policy kwargs
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({
            "features_extractor_class": CommonFeatureExtractor,
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
        print(f"Initialized GazePPO with use_gaze={config["model"]["use_gaze"]}, integration_method={config["model"]["feature_extractor"]}")

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