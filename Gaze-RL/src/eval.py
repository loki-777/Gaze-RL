## TODO

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f}")