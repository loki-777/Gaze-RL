import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Define a custom feature extractor that works with gaze
# Define a custom feature extractor for gaze integration
class SimpleGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        # Initialize the parent class first
        super().__init__(observation_space, features_dim)
        
        # Import your CustomCNN class
        from src.models.networks import CustomCNN
        
        # Create CNN with gaze support - hardcode use_gaze=True
        self.custom_cnn = CustomCNN(use_gaze=True)
        
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
            x = torch.cat([rgb, gaze], dim=1)  # (B, 4, H, W)
        else:
            x = rgb
        
        # Pass through CNN
        return self.custom_cnn(x)
    
    def set_gaze_heatmap(self, heatmap):
        self.gaze_heatmap = heatmap

class GazeFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor that processes RGB+Gaze input for RL agent.
    Compatible with Gymnasium.
    """
    def __init__(self, observation_space, features_dim=512, use_gaze=True):
        # Initialize parent
        super().__init__(observation_space, features_dim)
        
        # Determine input channels based on observation space
        self.use_gaze = use_gaze
        input_channels = 3  # Default to RGB
        
        # Check observation shape to determine channels
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 3:  # (H, W, C)
                input_channels = observation_space.shape[2]
                # Verify if we have 4 channels (RGB + gaze)
                if input_channels == 4 and use_gaze:
                    print("Using 4-channel input (RGB + gaze)")
                elif input_channels == 3 and use_gaze:
                    print("WARNING: Using gaze but observation space has only 3 channels")
                    # We'll handle this gracefully by not using gaze
                    self.use_gaze = False
        
        # Create CNN for feature extraction
        self.cnn = self._build_cnn(input_channels)
        
        # Output feature dimension
        self.features_dim = features_dim
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(features_dim)
        
    def _build_cnn(self, input_channels):
        """Build CNN architecture for feature extraction."""
        return nn.Sequential(
            # First convolutional layer
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Third convolutional layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten layer
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(64 * 9 * 9, 512),  # Size will depend on input dimensions
            nn.ReLU(),
            
            # Output layer
            nn.Linear(512, self.features_dim)
        )
        
    def forward(self, observations):
        """
        Process observations (images + optional gaze heatmap).
        
        Args:
            observations: Tensor of shape (batch_size, height, width, channels)
                         For RGB+gaze, channels should be 4
            
        Returns:
            torch.Tensor: Features extracted from observations
        """
        # Ensure observations have the right format for PyTorch (B, C, H, W)
        if isinstance(observations, np.ndarray):
            observations = torch.as_tensor(observations).float()
        
        # Normalize observations to [0, 1]
        if observations.max() > 1.0:
            observations = observations / 255.0
        
        # PyTorch expects (B, C, H, W) but SB3 uses (B, H, W, C)
        if observations.shape[-1] in [3, 4]:  # Last dim is channels
            observations = observations.permute(0, 3, 1, 2)
            
        # Forward pass through CNN
        features = self.cnn(observations)
        
        # Apply layer normalization
        features = self.layer_norm(features)
        
        return features


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