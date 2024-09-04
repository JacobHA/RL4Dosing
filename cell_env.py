import gymnasium as gym
import numpy as np
import random
from cell_model_pop_fde_slow_sde import Cell_Population
from gymnasium.wrappers import TimeLimit
from collections import deque

class CellEnv(gym.Env):
    def __init__(self, frame_stack=10, dt=0.1, alpha_mem=1, sigma=0.0, max_timesteps=100, **kwargs):
        self.dt = dt
        self.alpha_mem = alpha_mem
        # Use binary actions: apply antibiotic or not
        self.action_space = gym.spaces.Discrete(2)
        # Use continuous observations: cell population (or concentration)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(frame_stack,), dtype=np.float32)
        self.frame_stack = frame_stack
        # Initialize the cell population model
        T_final = max_timesteps * dt
        self.cell_population = Cell_Population(T_final, delta_t=dt, alpha_mem=alpha_mem, sigma=sigma, **kwargs)
        # self.previous_cost = None
        self.id = 'CellEnv-v0'
        # wrap in TimeLimit
        self.max_timesteps = max_timesteps
        self.step_count = 0

        self = TimeLimit(self, max_episode_steps=max_timesteps)


    def step(self, action):
        self.step_count += 1
        truncated = False
        terminated = False
        if self.step_count == self.max_timesteps:
            truncated = True
            terminated = True
            cost = 0
            tot = 1
            res_fraction = 0

        else:
            t, tot, N, R, cost = self.cell_population.simulate_population(action, delta_t=self.dt, plot=False)
            res_fraction = R / tot
            tot = tot[-1] # most recent cell count in entire simulation

        if tot >= 1000:
            terminated = True

        # Add to the state:
        self.stacked_states.append(cost)
        # Calculate reward
        reward = -cost

        return np.array(self.stacked_states, dtype=np.float32), reward, terminated, truncated, {'n_cells': tot, 'res_fraction': res_fraction}

    def reset(self, seed=None, h=2**(-7)):
        self.step_count = 0
        self.seed(seed)
        self.cell_population.initialize(h=h)
        state = 0.0
        n_cells = self.cell_population.init_conditions.sum()

        self.previous_cost = state
        self.stacked_states = deque([state] * self.frame_stack, maxlen=self.frame_stack)
        return np.array(self.stacked_states, dtype=np.float32), {'n_cells': n_cells}

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]