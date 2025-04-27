## TODO

from stable_baselines3 import PPO
from .networks import CustomCNN
from .gaze_predictor import GazePredictor
from ..utils import compute_gaze_iou, GradCAM
import torch

class GazePPO(PPO):
    def __init__(self, config):
        policy_kwargs = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"use_gaze": config["use_gaze"]}
        }
        super().__init__("CnnPolicy", env, **policy_kwargs)
        
        self.config = config
        
        # Gaze predictor
        self.gaze_predictor = GazePredictor()
        self.gaze_predictor.load_state_dict(torch.load(config["gaze_predictor_path"]))
        self.gaze_predictor.eval()
        
        #TODO GradCAM extractor and you need to specify which layer you want
        self.gradcam_extractor = GradCAM(
            self.policy.features_extractor.backbone,
            self.policy.features_extractor.backbone.layer4
        )
    def compute_gaze_reward(self, obs_tensor):
        """
        obs_tensor: torch.Tensor (1, C, H, W)
        """
        with torch.no_grad():
            gradcam_map = self.gradcam_extractor.generate(obs_tensor)
            gaze_map = self.gaze_predictor(obs_tensor)
            #TODO need to aligh gradcam and gaze size
            gaze_map = torch.nn.functional.interpolate(
                gaze_map, size=gradcam_map.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze()

        gaze_iou = compute_gaze_iou(gradcam_map, gaze_map)
        r_gaze = self.config["lambda_gaze"] * gaze_iou

        return r_gaze
        