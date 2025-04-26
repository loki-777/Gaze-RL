## TODO

from stable_baselines3 import PPO
from .networks import CustomCNN
from .gaze_predictor import GazePredictor
from ..utils import compute_gaze_iou, compute_reward, compute_metrics
class GazePPO(PPO):
    def __init__(self, config):
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"use_gaze": config["use_gaze"]}
        }
        super().__init__("CnnPolicy", env, **policy_kwargs)
        
        self.config = config
        
        # Load gaze predictor from checkpoint
        self.gaze_predictor = GazePredictor()
        self.gaze_predictor.load_state_dict(torch.load(config["gaze_predictor_path"]))
        # gaze prediction 
        # grad_cam
        grad_cam_prediction = #TODO
        gaze_prediction = self.gaze_predictor(obj)
        gaze_iou = compute_gaze_iou(grad_cam_prediction, gaze_prediction) # TODO need to be checked for imgs size
        reward = compute_reward(gaze_iou, config["gaze_iou_threshold"])
        metrics = compute_metrics(reward, config["reward_threshold"])
        