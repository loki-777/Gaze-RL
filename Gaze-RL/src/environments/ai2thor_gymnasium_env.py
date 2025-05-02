import gymnasium as gym
import numpy as np
import ai2thor.controller
from gymnasium import spaces
import random
import cv2

class AI2ThorEnv(gym.Env):
    """AI2-THOR Environment for Object Search Tasks compatible with Gymnasium"""
    
    # Metadata for environment rendering
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, config, render_mode=None):
        """Initialize AI2-THOR environment.
        
        Args:
            config: Dictionary with environment configuration
            render_mode: Rendering mode (optional)
        """
        self.config = config
        
        # Validate render mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}. Must be one of {self.metadata['render_modes']}")
            
        self.render_mode = render_mode
        
        # Initialize AI2-THOR controller
        self.controller = ai2thor.controller.Controller(
            width=config.get("width", 224),
            height=config.get("height", 224),
            gridSize=config.get("grid_size", 0.25),
            fieldOfView=config.get("fov", 90),
            renderDepthImage=config.get("depth", True),
            renderInstanceSegmentation=config.get("segmentation", True),
        )
        
        # Define action space: Move (4 directions), Rotate (2 directions), Look (up/down)
        self.action_space = spaces.Discrete(7)
        
        # Define observation space: RGB image (224x224x3)
        h = config.get("height", 224)
        w = config.get("width", 224)
        
        # For compatibility with SB3, we'll use a simple Box space for RGB images
        # You can expand this to Dict space if needed and use appropriate wrappers
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(h, w, 3), 
            dtype=np.uint8
        )
        
        # Set target object
        self.target_object = config.get("target_object", "Microwave")
        
        # Track visited positions
        self.visited_positions = set()
        
        # Episode tracking
        self.steps = 0
        self.max_steps = config.get("max_steps", 200)
        
    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Additional options for environment reset
            
        Returns:
            tuple: (observation, info)
        """
        attempts = 0
        max_attempts = 3

        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset episode variables
        self.steps = 0
        self.visited_positions = set()
        
        while attempts < max_attempts:
            try:
                problematic_scenes = ["FloorPlan3", "FloorPlan26"]
                # Select random scene from list of kitchen scenes
                kitchen_scenes = [f"FloorPlan{i}" for i in range(1, 31) if i <= 5 or 25 <= i <= 30]
                kitchen_scenes = [s for s in kitchen_scenes if s not in problematic_scenes]
                #kitchen_scenes = ["FloorPlan30"]
                scene = random.choice(kitchen_scenes)
                
                # Initialize scene
                self.controller.reset(scene)
                self.controller.step(dict(action="Initialize", gridSize=self.config.get("grid_size", 0.25)))
                
                # Randomize agent starting position
                reachable_positions = self.controller.step(dict(action="GetReachablePositions")).metadata["actionReturn"]
                if reachable_positions:
                    random_position = random.choice(reachable_positions)
                    self.controller.step(dict(
                        action="Teleport",
                        position=dict(
                            x=random_position["x"],
                            y=random_position["y"],
                            z=random_position["z"]
                        )
                    ))
                
                # Get initial observation
                obs = self._get_observation()
                
                # Add current position to visited positions
                agent_pos = self._get_agent_position_key()
                self.visited_positions.add(agent_pos)
                
                # Additional info dictionary
                info = {
                    "scene": scene,
                    "visited_positions": len(self.visited_positions),
                    "agent_position": agent_pos,
                }
                
                # Render if needed
                if self.render_mode == "human":
                    self.render()
                
                # Return observation and info
                return obs["rgb"], info
            
            except TimeoutError:
                attempts += 1
                print(f"Timeout during reset, attempt {attempts}/{max_attempts}")
                
                # Recreate controller if needed
                if self.controller:
                    try:
                        self.controller.stop()
                    except:
                        pass
                
                self.controller = ai2thor.controller.Controller(
                    width=self.config.get("width", 224),
                    height=self.config.get("height", 224),
                    gridSize=self.config.get("grid_size", 0.25),
                    fieldOfView=self.config.get("fov", 90),
                    renderDepthImage=self.config.get("depth", True),
                    renderInstanceSegmentation=self.config.get("segmentation", True),
                )
        
        # If we've exhausted attempts, raise the error
        raise RuntimeError("Failed to reset environment after multiple attempts")
        
    def step(self, action):
        """Take action in environment.
        
        Args:
            action: Integer action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Increment step counter
        self.steps += 1
        
        # Convert action index to AI2-THOR action
        thor_action = self._get_thor_action(action)
        
        # Check if action is a repeat
        prev_position = self._get_agent_position_key()
        
        # Execute action
        event = self.controller.step(thor_action)
        
        # Get current position
        current_position = self._get_agent_position_key()
        repeated_action = current_position == prev_position and thor_action["action"] in ["MoveAhead"]
        
        # Add current position to visited positions
        self.visited_positions.add(current_position)
        
        # Get observation
        obs = self._get_observation()
        
        # Check if object is visible
        visible_objects = event.metadata["objects"]
        object_visible = False
        object_distance = float('inf')
        object_visibility = 0.0
        
        target_objects = [obj for obj in visible_objects if obj["objectType"] == self.target_object and obj["visible"]]
        if target_objects:
            object_visible = True
            # Get distance to closest target object
            object_distance = min([self._get_distance_to_object(obj) for obj in target_objects])
            # Calculate object visibility (percentage of pixels)
            if hasattr(event, "instance_masks") and event.instance_masks is not None:
                masks = [mask for obj_id, mask in event.instance_masks.items() 
                         if any(obj["objectId"] == obj_id for obj in target_objects)]
                if masks:
                    # Calculate visibility as percentage of pixels
                    total_pixels = event.frame.shape[0] * event.frame.shape[1]
                    object_pixels = sum(mask.sum() for mask in masks)
                    object_visibility = object_pixels / total_pixels
        
        # Check for success condition
        success = object_visible and object_distance < 1.5 and object_visibility > 0.05
        
        # Calculate reward
        reward = self._compute_reward(
            object_visible=object_visible,
            object_distance=object_distance,
            object_visibility=object_visibility,
            repeated_action=repeated_action,
            success=success
        )
        
        # Check termination conditions
        terminated = success
        truncated = self.steps >= self.max_steps
        
        # Prepare info dictionary
        info = {
            "success": success,
            "steps": self.steps,
            "object_visible": object_visible,
            "object_distance": object_distance,
            "object_visibility": object_visibility,
            "repeated_action": repeated_action,
            "visited_positions": len(self.visited_positions),
            "exploration_coverage": len(self.visited_positions) / 100.0,  # Normalize by approximate reachable positions
            "visible_objects": [obj["objectType"] for obj in visible_objects if obj["visible"]],
        }
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return obs["rgb"], reward, terminated, truncated, info
    
    def _get_observation(self):
        """Extract observation from AI2-THOR event."""
        frame = self.controller.last_event.frame
        depth = self.controller.last_event.depth_frame
        instance_seg = self.controller.last_event.instance_segmentation_frame
        
        # Ensure depth has correct shape
        if depth is not None:
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
        else:
            depth = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            
        # Ensure instance segmentation has correct shape
        if instance_seg is None:
            instance_seg = np.zeros_like(frame)
        
        return {
            "rgb": frame,
            "depth": depth,
            "segmentation": instance_seg
        }
    
    def _get_thor_action(self, action_idx):
        """Convert action index to AI2-THOR action."""
        actions = {
            0: dict(action="MoveAhead"),
            1: dict(action="MoveRight"),
            2: dict(action="MoveLeft"),
            3: dict(action="MoveBack"),
            4: dict(action="RotateRight"),
            5: dict(action="RotateLeft"),
            6: dict(action="LookUp", degrees=30),
            # 7: dict(action="LookDown", degrees=30)
        }
        return actions.get(action_idx, dict(action="Pass"))
    
    def _get_agent_position_key(self):
        """Get hashable representation of agent position."""
        metadata = self.controller.last_event.metadata
        x = round(metadata["agent"]["position"]["x"], 2)
        z = round(metadata["agent"]["position"]["z"], 2)
        rot = round(metadata["agent"]["rotation"]["y"] / 90.0) % 4
        return (x, z, rot)
    
    def _get_distance_to_object(self, obj):
        """Calculate distance between agent and object."""
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        obj_pos = obj["position"]
        
        return np.sqrt(
            (agent_pos["x"] - obj_pos["x"])**2 +
            (agent_pos["y"] - obj_pos["y"])**2 +
            (agent_pos["z"] - obj_pos["z"])**2
        )
    
    def _compute_reward(self, object_visible, object_distance, object_visibility, repeated_action, success):
        """Compute reward based on task progress."""
        reward = -0.01  # Small step penalty
        
        # Penalty for repeated actions
        if repeated_action:
            reward -= 0.1
        
        # Reward for finding object
        if object_visible:
            # Distance-based reward component
            distance_reward = 1.0 / (1.0 + object_distance)
            
            # Visibility-based reward component
            visibility_reward = object_visibility
            
            # Combined reward for object detection
            reward += distance_reward + visibility_reward
        
        # Success reward
        if success:
            reward += 10.0
        
        return reward
    
    def render(self):
        """Render current frame according to render_mode."""
        if self.render_mode is None:
            return None
            
        if self.render_mode == "human":
            # Display the frame using OpenCV
            cv2.imshow('AI2-THOR', self.controller.last_event.frame)
            cv2.waitKey(1)
            return None
            
        elif self.render_mode == "rgb_array":
            # Return the RGB frame
            return self.controller.last_event.frame
    
    def close(self):
        """Clean up environment resources."""
        try:
            # Explicitly stop controller
            if hasattr(self, 'controller') and self.controller is not None:
                self.controller.stop()
                self.controller = None
                
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error stopping controller: {e}")
            
        # Close any open windows
        cv2.destroyAllWindows()