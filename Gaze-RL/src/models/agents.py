## TODO

from stable_baselines3 import PPO
from .networks import CustomCNN

class GazePPO(PPO):
    def __init__(self, config):
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"use_gaze": config["use_gaze"]}
        }
        super().__init__("CnnPolicy", env, **policy_kwargs)