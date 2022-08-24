# imports
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from Korero.envs_korero import AT_env_korero

# define paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'SavedModels', 'PP0_AT_linRew_competition3_entcoeff0_01_2e5steps_2')


# create environment
env = AT_env_korero()

# create model
model = PPO('MlpPolicy', env, tensorboard_log=log_path, ent_coef=0.01)

# train model
model.learn(total_timesteps=int(2e5))

# save model
model.save(model_path)

# evaluate model
evaluate_policy(model, env, n_eval_episodes=5, render=False)
env.close()