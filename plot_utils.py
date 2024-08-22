import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
from cell_env import CellEnv
import multiprocessing as mp
from stable_baselines3 import DQN


def run_episode(env_args, model_str=None):
    # Initialize the environment
    env = CellEnv(**env_args)
    if model_str is not None:
        model = DQN.load(model_str)

    done = False
    obs, _ = env.reset()
    episode_observations = []
    action_sequence = []
    while not done:
        if model_str is not None:
            action, _states = model.predict(obs)
        else:
            action = env.action_space.sample()
        obs, rewards, term, trunc, info = env.step(action)
        done = term or trunc
        if not done:
            episode_observations.append(info['n_cells'])
            action_sequence.append(action)
    
    return episode_observations, action_sequence

def evaluate_model(env_args, num_episodes, model_str=None, multiprocess=False):
    """
    Evaluate the model over several episodes and plot the results.

    Parameters:
    - model: Trained RL model to be evaluated.
    - eval_env: The evaluation environment instance.
    - num_episodes: Number of episodes to evaluate.
    - multiprocess: Boolean flag to enable multiprocessing.

    Returns:
    - avg_observations: Average observations at each step.
    - all_observations: List of observations for all episodes.
    """
    all_observations = []
    all_actions = []

    if multiprocess:
        with mp.Pool(mp.cpu_count() - 1) as pool:
            results = pool.starmap(run_episode, [(env_args, model_str)] * num_episodes)
   
        # Unpack the results
        all_observations, all_actions = zip(*results)
    
    else:
        # load the model and env:
        eval_env = CellEnv(**env_args)
        model = DQN.load(model_str)

        for _ in range(num_episodes):
            obs, actions = run_episode(eval_env, model)
            all_observations.append(obs)
            all_actions.append(actions)

    # convert to numpy arrays
    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)
    return all_observations, all_actions


def plot_observations(env_args: dict, 
                      all_observations: np.ndarray,
                      all_unif_obs=None,
                      alpha_val=0.05,
                      log_axis='',
                      ):
    """
    Plot the average observations and individual episode tracks.

    Parameters:
    - avg_observations: Average observations at each step.
    - all_observations: List of observations for all episodes.
    """
    plt.figure(figsize=(12, 8))
    dt = env_args['dt']
    max_timesteps = env_args['max_timesteps']
    x_axis = np.linspace(0, dt*max_timesteps, max_timesteps-1)
    
    # Plot individual episode tracks with lower alpha
    for obs in all_observations:
        plt.plot(x_axis, obs, alpha=alpha_val, linewidth=0.5, color='black')

    if all_unif_obs is not None:
        for obs in all_unif_obs:
            plt.plot(x_axis, obs, alpha=alpha_val, linewidth=0.5, color='red')
    
    # Plot average observations
    avg_observations = all_observations.mean(axis=0)
    plt.plot(x_axis, avg_observations, label='trained policy', linewidth=3, color='black')
    if all_unif_obs is not None:
        unif_obs = all_unif_obs.mean(axis=0)
        plt.plot(x_axis, unif_obs, label='random policy', linewidth=3, color='red')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Total cells (n_cells)')
    plt.title('Model Evaluation: Average Observations and Individual Episode Tracks')
    plt.legend()
    if 'x' in log_axis:
        plt.xscale('log')
    if 'y' in log_axis:
        plt.yscale('log')
    plt.show()

def plot_actions(env_args: dict,
                all_actions: np.ndarray,
                alpha_val=0.05,
                log_axis='',
                ):
    """
    Plot the average actions and individual episode tracks.
    """

    plt.figure(figsize=(12, 8))
    dt = env_args['dt']
    max_timesteps = env_args['max_timesteps']
    x_axis = np.linspace(0, dt*max_timesteps, max_timesteps-1)

    # Plot individual episode tracks with lower alpha
    for actions in all_actions:
        plt.plot(x_axis, actions, alpha=alpha_val, linewidth=0.5, color='black')
    
    # plot mean actions:
    avg_actions = all_actions.mean(axis=0)
    plt.plot(x_axis, avg_actions, label='mean actions', linewidth=3, color='black')

    plt.xlabel('Time (hours)')
    plt.ylabel('Action')
    plt.title('Model Evaluation: Average Actions and Individual Episode Tracks')
    if 'x' in log_axis:
        plt.xscale('log')
    if 'y' in log_axis:
        plt.yscale('log')
    plt.show()