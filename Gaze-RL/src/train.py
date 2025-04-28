# ## TODO

# ##### NOT TO BE USED

# from src.environments import AI2ThorEnv
# from src.models.agents import GazePPO
# from configs import default

# # env = AI2ThorEnv(default.ENV_CONFIG)
# # model = GazePPO(default.MODEL_CONFIG)
# # model.learn(total_timesteps=100000)


# from src.environments.ai2thor_env import AI2ThorEnv
# from src.models.agents import GazePPO
# from configs import default
# from .utils import compute_reward, compute_metrics
# import numpy as np
# import torch

# def main():
#     env = AI2ThorEnv(default.ENV_CONFIG)
#     model = GazePPO(default.MODEL_CONFIG, env)

#     obs = env.reset()
#     done = False
#     total_reward = 0

#     episode_rewards = []
#     success_flags = []
#     steps_list = []

#     current_ep_reward = 0
#     steps = 0

#     while not done:
#         # obs to tensor
#         obs = np.transpose(obs, (2, 0, 1))
#         obs = torch.from_numpy(obs).float()
        
#         # action predict
#         action, _ = model.predict(obs)

#         # gain env reward
#         next_obs, reward_env, done, info = env.step(action)

#         # gaze reward
#         r_gaze = model.compute_gaze_reward(obs_tensor)

#         #TODO total reward
#         success = info.get("success", 0) #?
#         step_taken = steps
#         progress_delta = info.get("progress_delta", 0.0) #??

#         # gaze rewardと環境rewardを合成
#         total_step_reward = compute_reward(
#             success=success,
#             step_taken=step_taken,
#             gaze_iou=r_gaze / model.config["lambda_gaze"],
#             progress_delta=progress_delta
#         )
        
#         current_ep_reward += total_step_reward
#         steps += 1

#         obs = next_obs

#         if done:
#             episode_rewards.append(current_ep_reward)
#             success_flags.append(success)
#             steps_list.append(steps)

#             # next episode
#             obs = env.reset()
#             current_ep_reward = 0
#             steps = 0

#         current_ep_reward += total_step_reward
#         steps += 1

#     print("Episode finished.")
#     metrics = compute_metrics(episode_rewards, success_flags, steps_list)
#     print(metrics)

# if __name__ == "__main__":
#     main()