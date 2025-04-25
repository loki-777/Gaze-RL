import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to flatten Dict observation spaces for compatibility with Stable-Baselines3.
    Works with Gymnasium environments.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Assuming env.observation_space is a Dict
        if isinstance(env.observation_space, spaces.Dict):
            spaces_dict = env.observation_space.spaces
            
            # For now, we'll only use the RGB image component
            # This is a simplified approach - using only RGB part of the observation
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=spaces_dict["rgb"].shape,
                dtype=np.uint8
            )
        # If already a Box space, keep it
        elif isinstance(env.observation_space, spaces.Box):
            self.observation_space = env.observation_space
    
    def observation(self, observation):
        # If observation is a dict, extract RGB component
        if isinstance(observation, dict):
            return observation["rgb"]
        # If already the right format, just return it
        return observation


class DictToTupleWrapper(gym.ObservationWrapper):
    """
    Alternative wrapper that converts Dict observations to a tuple of observations.
    This is useful if you want to keep all observation components.
    Compatible with Gymnasium.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Assuming env.observation_space is a Dict
        if isinstance(env.observation_space, spaces.Dict):
            spaces_dict = env.observation_space.spaces
            
            # Create a tuple space from the individual observation spaces
            self.observation_space = spaces.Tuple(tuple(spaces_dict.values()))
            
            # Keep track of the original keys
            self.keys = list(spaces_dict.keys())
    
    def observation(self, observation):
        # If observation is a dict, convert to tuple
        if isinstance(observation, dict):
            # Convert dictionary to tuple in the same order as self.keys
            return tuple(observation[key] for key in self.keys)
        # If already a tuple, just return it
        return observation
        
        
class GazeEnvWrapper(gym.Wrapper):
    """
    Wrapper to add gaze prediction capabilities to an environment.
    Compatible with Gymnasium.
    """
    def __init__(self, env, gaze_predictor=None):
        super().__init__(env)
        self.gaze_predictor = gaze_predictor
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.gaze_predictor:
            # Predict gaze heatmap
            gaze_heatmap = self._predict_gaze(obs)
            
            # Add gaze heatmap to info
            info["gaze_heatmap"] = gaze_heatmap
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.gaze_predictor:
            # Predict gaze heatmap
            gaze_heatmap = self._predict_gaze(obs)
            
            # Add gaze heatmap to info
            info["gaze_heatmap"] = gaze_heatmap
            
            # You could augment reward here using the gaze info if needed
            # For example:
            # reward = self._augment_reward(reward, obs, action, gaze_heatmap)
        
        return obs, reward, terminated, truncated, info
    
    def _predict_gaze(self, image):
        """Predict gaze heatmap from RGB image."""
        import torch
        import cv2
        
        # Resize if needed
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        else:
            # Assume it's already a tensor
            image_tensor = image
        
        # Run prediction
        with torch.no_grad():
            heatmap = self.gaze_predictor(image_tensor).squeeze().cpu().numpy()
        
        return heatmap
    
    def _augment_reward(self, reward, obs, action, gaze_heatmap):
        """Augment reward based on gaze information."""
        # Example implementation - you can customize this
        
        # Extract agent's current view coordinates (from middle of frame)
        h, w = gaze_heatmap.shape
        agent_position = (w//2, h//2)
        
        # Calculate attention at agent position
        attention_value = gaze_heatmap[agent_position[1], agent_position[0]]
        
        # Add small bonus based on attention value
        reward += 0.1 * attention_value
        
        return reward