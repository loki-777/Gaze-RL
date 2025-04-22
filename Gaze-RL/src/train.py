## TODO

from src.environments import AI2ThorEnv
from src.models.agents import GazePPO
from configs import default

env = AI2ThorEnv(default.ENV_CONFIG)
model = GazePPO(default.MODEL_CONFIG)
model.learn(total_timesteps=100000)