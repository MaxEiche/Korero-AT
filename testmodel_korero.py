import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Korero.envs_korero import AT_env_korero
from stable_baselines3.common.monitor import Monitor


def plot_trajectory(current_ax):
    current_ax.fill_between(env.x, 0, 100 * env.trajectory[2, :], color='orange', label='drug')
    current_ax.plot(env.x, env.trajectory[0, :], 'b', label='wt')
    current_ax.plot(env.x, env.trajectory[1, :], 'g', label='mut')
    current_ax.plot(env.x, env.trajectory[0, :] + env.trajectory[1, :], 'k', label='total', linewidth=2)
    current_ax.set_xlabel('time [steps]')
    current_ax.set_ylabel('burden [a.u.]')


def fixed_strategy(observation, treatment_threshold=90):
    if observation[0]+observation[1] > treatment_threshold:
        return 1
    else:
        return 0


# load environment
env = AT_env_korero()
env = Monitor(env)

# load the model
model_name = 'PP0_AT_1e5steps'
model_path = os.path.join('Training', 'SavedModels', model_name)
model = PPO.load(model_path, env=env)

# Test the environment
episodes = 12
obs = env.reset()

# Initialize some things
final_score = np.zeros(episodes)

# initialize plotting
numrows = np.floor(np.sqrt(episodes)).astype('int')
numcols = np.ceil(episodes/numrows).astype('int')
fig, ax = plt.subplots(numrows, numcols)
ax = np.reshape(ax, -1)

for episode in range(episodes):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    final_score[episode] = score
    print(f'Episode{episode} - Score:{score}')

    plot_trajectory(ax[episode])

plt.show()
print(f'mean reward: {np.mean(final_score)} +/- {np.std(final_score)}')

env.close()
