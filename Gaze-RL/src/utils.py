# should contain helper functions for rewards, loggers, wrappers
## TODO

def calculate_reward(obs, action, gaze_heatmap=None):
    reward = -0.01  # Step penalty
    if gaze_heatmap:
        reward += 0.1 * iou(agent_attention, gaze_heatmap)
    return reward