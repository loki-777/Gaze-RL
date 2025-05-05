import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces, Wrapper
from typing import Dict, Tuple, Any, Optional
import cv2
import os
from datetime import datetime
import time

class RetryTimeoutWrapper(Wrapper):
    """Wrapper to handle TimeoutError and retry actions"""
    
    def __init__(self, env, max_retries=3, retry_delay=1.0):
        super().__init__(env)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def step(self, action):
        for attempt in range(self.max_retries):
            try:
                return self.env.step(action)
            except TimeoutError as e:
                if attempt < self.max_retries - 1:
                    print(f"TimeoutError encountered, retrying action (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Max retries exceeded, returning empty observation with termination")
                    # Create a zero-filled observation with the proper shape
                    zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                    # Return a terminal state with a small penalty
                    return zero_obs, -1.0, True, True, {"timeout": True}
class VideoRecorderWrapper(Wrapper):
    """Records episodes as videos."""
    
    def __init__(self, env, video_dir="videos", record_freq=5, video_quality=0.95):
        """
        Initialize the video recorder wrapper.
        Args:
            env: The environment to wrap
            video_dir: Directory to save videos
            record_freq: Record every N-th episode (default: 5)
            video_quality: Quality of the video (0.0-1.0, higher is better)
        """
        super().__init__(env)
        self.video_dir = video_dir
        self.record_freq = record_freq
        self.video_quality = video_quality
        self.frames = []
        self.episode_count = 0
        self.recording = False
        # Create video directory if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Increment episode counter and determine if we should record
        self.episode_count += 1
        self.recording = (self.episode_count % self.record_freq == 0)

        if self.recording:
            print(f"Recording episode {self.episode_count}...")
            # Clear frames buffer at start of episode
            self.frames = []
            # Capture the initial frame (handle both dict and array observations)
            if hasattr(self.env.unwrapped, 'last_image'):
                # Direct access to the AI2Thor last rendered image
                frame = self.env.unwrapped.last_image.copy()
                self.frames.append(frame)
            elif isinstance(obs, dict) and "rgb" in obs:
                frame = obs["rgb"].copy()
                self.frames.append(frame)
            elif len(obs.shape) == 3 and obs.shape[2] == 3:
                # Simple RGB observation
                frame = obs.copy()
                self.frames.append(frame)
            else:
                # Observation might be flattened or have gaze channel
                # Try to extract RGB information
                try:
                    # For flattened observations with known shape
                    rgb_obs = obs.reshape(224, 224, 4)[:, :, :3].copy()
                    self.frames.append(rgb_obs)
                except:
                    print("Warning: Could not capture frame for video recording")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Only capture frame if we're recording this episode
        if self.recording:
            # Capture frame (handle both dict and array observations)
            if hasattr(self.env.unwrapped, 'last_image'):
                # Direct access to the AI2Thor last rendered image
                frame = self.env.unwrapped.last_image.copy()
                self.frames.append(frame)
            elif isinstance(obs, dict) and "rgb" in obs:
                frame = obs["rgb"].copy()
                self.frames.append(frame)
            elif len(obs.shape) == 3 and obs.shape[2] == 3:
                # Simple RGB observation
                frame = obs.copy()
                self.frames.append(frame)
            else:
                # Observation might be flattened or have gaze channel
                # Try to extract RGB information
                try:
                    # For flattened observations with known shape
                    rgb_obs = obs.reshape(224, 224, 4)[:, :, :3].copy()
                    self.frames.append(rgb_obs)

                except:
                    pass  # Skip this frame if we can't process it

            # Save video if episode is done
            if (terminated or truncated) and self.frames:
                self.save_video()
        return obs, reward, terminated, truncated, info
    
    def save_video(self):
        """Save recorded frames as a video file."""
        if not self.frames or len(self.frames) < 2:
            print("Warning: Not enough frames to create video")
            return
            
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
        
        try:
            # Get frame dimensions
            height, width = self.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10  # Frames per second
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in self.frames:
                # Ensure frame has right dimensions
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Convert to BGR (OpenCV format) if needed
                if frame.shape[-1] == 3:  # RGB format
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                out.write(frame)
            
            # Release video writer
            out.release()
            print(f"Saved video with {len(self.frames)} frames to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
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


class GazeEnvWrapper(gym.Wrapper):
    """
    Enhanced wrapper to add gaze prediction capabilities to an environment.
    Compatible with Gymnasium and SB3.
    
    This wrapper:
    1. Predicts gaze heatmaps from visual observations
    2. Stores gaze heatmaps in info dict for use by RL agent
    3. Can optionally augment rewards based on gaze information
    """
    def __init__(self, env, gaze_predictor=None, augment_reward=True):
        super().__init__(env)
        self.gaze_predictor = gaze_predictor
        self.augment_reward = augment_reward
        
        # Store the last predicted gaze heatmap
        self.last_gaze_heatmap = None
        
        # Flag to ensure the gaze model is in eval mode
        if self.gaze_predictor is not None and hasattr(self.gaze_predictor, 'eval'):
            self.gaze_predictor.eval()
    
    def reset(self, **kwargs):
        """Reset environment and predict initial gaze heatmap."""
        obs, info = self.env.reset(**kwargs)
        
        if self.gaze_predictor is not None:
            # Get RGB image from observation
            image = self._get_rgb_image(obs)
            
            # Predict gaze heatmap
            gaze_heatmap = self._predict_gaze(image)
            self.last_gaze_heatmap = gaze_heatmap
            
            # Add gaze heatmap to info
            info["gaze_heatmap"] = gaze_heatmap
        
        return obs, info
    
    def step(self, action):
        """Step environment and update gaze heatmap."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.gaze_predictor is not None:
            # Get RGB image from observation
            image = self._get_rgb_image(obs)
            
            # Predict gaze heatmap
            gaze_heatmap = self._predict_gaze(image)
            self.last_gaze_heatmap = gaze_heatmap
            
            # Add gaze heatmap to info
            info["gaze_heatmap"] = gaze_heatmap
            
            # Augment reward using the gaze info if enabled
            if self.augment_reward:
                reward = self._augment_reward(reward, obs, action, gaze_heatmap, info)
        
        return obs, reward, terminated, truncated, info
    
    def _get_rgb_image(self, observation):
        """Extract RGB image from observation."""
        # If observation is a dict, extract RGB component
        if isinstance(observation, dict) and "rgb" in observation:
            image = observation["rgb"]
        else:
            # Assume observation is already the RGB image
            image = observation
        
        return image
    
    def _predict_gaze(self, image):
        """Predict gaze heatmap from RGB image."""
        # Ensure model is in eval mode
        if hasattr(self.gaze_predictor, 'eval') and not self.gaze_predictor.training:
            self.gaze_predictor.eval()
            
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # Resize if needed
            if image.shape[:2] != (224, 224):
                import cv2
                image = cv2.resize(image, (224, 224))
            
            # Normalize image
            if image.max() > 1.0:
                image = image / 255.0
                
            # Convert to tensor and add batch dimension
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
            
            # Move to same device as model
            if hasattr(self.gaze_predictor, 'device'):
                image_tensor = image_tensor.to(self.gaze_predictor.device)
            
            # Run prediction
            with torch.no_grad():
                # Different model types might have different forward methods
                try:
                    if hasattr(self.gaze_predictor, 'predict_step'):
                        # PyTorch Lightning model
                        heatmap = self.gaze_predictor.predict_step(image_tensor, None)
                    else:
                        # Standard PyTorch model
                        heatmap = self.gaze_predictor(image_tensor)
                    
                    # Extract heatmap data
                    if isinstance(heatmap, torch.Tensor):
                        heatmap = heatmap.squeeze().cpu().numpy()
                    
                    return heatmap
                except Exception as e:
                    print(f"Error predicting gaze: {e}")
                    # Return empty heatmap in case of error
                    return np.zeros((224, 224), dtype=np.float32)
        
        # If input is already a tensor, just return it
        return image
    
    def _augment_reward(self, reward, obs, action, gaze_heatmap, info):
        """Augment reward based on gaze information and object visibility."""
        # Base reward remains the same
        augmented_reward = reward
        
        # Only add gaze-based bonus if the target object is not already visible
        if not info.get("object_visible", False):
            # Extract agent's current view coordinates (from middle of frame)
            h, w = gaze_heatmap.shape if isinstance(gaze_heatmap, np.ndarray) else (224, 224)
            
            # Determine object visibility regions from segmentation if available
            object_regions = info.get("object_regions", [])
            
            # Calculate gaze-object overlap
            attention_score = 0.0
            
            if object_regions:
                # If we have object segmentation, calculate overlap between gaze and objects
                for region in object_regions:
                    # Calculate IoU between gaze heatmap and object region
                    region_mask = np.zeros((h, w), dtype=np.float32)
                    x1, y1, x2, y2 = region  # Assuming region is [x1, y1, x2, y2]
                    region_mask[y1:y2, x1:x2] = 1.0
                    
                    # Threshold gaze heatmap
                    gaze_binary = (gaze_heatmap > 0.2).astype(np.float32)
                    
                    # Calculate IoU
                    intersection = np.logical_and(region_mask, gaze_binary).sum()
                    union = np.logical_or(region_mask, gaze_binary).sum()
                    
                    if union > 0:
                        attention_score = max(attention_score, intersection / union)
            else:
                # If no object regions, use center of gaze as attention metric
                # Find peak attention location
                if isinstance(gaze_heatmap, np.ndarray):
                    peak_y, peak_x = np.unravel_index(gaze_heatmap.argmax(), gaze_heatmap.shape)
                    # Calculate distance from center of frame
                    center_y, center_x = h // 2, w // 2
                    distance = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
                    max_distance = np.sqrt(h**2 + w**2) / 2
                    
                    # Normalize and invert distance to get attention score
                    attention_score = 1.0 - min(1.0, distance / max_distance)
                    
                    # Get max value from heatmap as confidence
                    confidence = gaze_heatmap.max()
                    
                    # Combine with confidence
                    attention_score = attention_score * confidence
            
            # Add small reward based on attention
            # Scale is small to avoid overpowering main reward signal
            gaze_reward = 0.1 * attention_score
            augmented_reward += gaze_reward
            
            # Store the gaze reward component for analysis
            info["gaze_reward"] = gaze_reward
        
        return augmented_reward


class GazePreprocessEnvWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that adds the gaze heatmap as additional channel to RGB observations.
    This creates 4-channel observations: RGB + gaze heatmap.
    Compatible with Gymnasium and SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Check if original observation space is Box
        if isinstance(self.env.observation_space, spaces.Box):
            # Get original shape
            original_shape = self.env.observation_space.shape
            
            if len(original_shape) == 3:  # (H, W, C) format
                h, w, c = original_shape
                
                # Create new observation space with additional channel
                self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, c + 1),  # Add one channel for gaze
                    dtype=np.uint8
                )
            else:
                # If shape doesn't match expected format, keep original space
                self.observation_space = self.env.observation_space
        else:
            # If not a Box space, keep original space
            self.observation_space = self.env.observation_space
    
    def observation(self, observation):
        """Add gaze heatmap as additional channel to RGB observation."""
        # Check if we have a gaze heatmap in the info dict
        gaze_heatmap = None
        if hasattr(self.env, 'last_gaze_heatmap'):
            gaze_heatmap = self.env.last_gaze_heatmap
        
        # If no gaze heatmap available, return original observation
        if gaze_heatmap is None:
            return observation
        
        # Ensure gaze_heatmap has correct shape and type
        if isinstance(gaze_heatmap, np.ndarray):
            # Resize gaze heatmap to match observation shape
            if gaze_heatmap.shape[:2] != observation.shape[:2]:
                import cv2
                gaze_heatmap = cv2.resize(
                    gaze_heatmap, 
                    (observation.shape[1], observation.shape[0])
                )
            
            # Normalize and convert to uint8
            gaze_uint8 = (gaze_heatmap * 255).astype(np.uint8)
            
            # Expand dimensions to match observation
            if len(gaze_uint8.shape) == 2:  # (H, W) -> (H, W, 1)
                gaze_uint8 = gaze_uint8[..., np.newaxis]
            
            # Concatenate observation and gaze heatmap
            combined_obs = np.concatenate([observation, gaze_uint8], axis=-1)
            
            return combined_obs
        
        # If gaze_heatmap is not a valid array, return original observation
        return observation