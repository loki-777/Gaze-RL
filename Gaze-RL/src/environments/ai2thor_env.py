## TODO

import gym
import ai2thor.controller

class AI2ThorEnv(gym.Env):
    def __init__(self, config):
        self.controller = ai2thor.controller.Controller()
        self.action_space = gym.spaces.Discrete(7)  # Custom actions
        self.observation_space = gym.spaces.Box(...)  # RGB image
    
    def step(self, action):
        # Implement action logic, reward calculation
        return obs, reward, done, info
    
    def reset(self):
        return obs