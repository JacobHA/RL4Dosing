import gymnasium as gym
import numpy as np
import random
from cell_model_pop_fde_slow_sde import Cell_Population
from gymnasium.wrappers import TimeLimit

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
        if self.step_count == self.max_timesteps:
            truncated = True
            cost = 0
            tot = None
        else:
            t, tot, cost = self.cell_population.simulate_population(action, delta_t=self.dt, plot=False)
            tot = np.mean(tot)

        # todo: make this a deque:
        self.stacked_states.pop(0)
        self.stacked_states.append(cost)
        # Calculate reward
        reward = -cost

        return np.array(self.stacked_states, dtype=np.float32), reward, False, truncated, {'n_cells': tot}

    def reset(self, seed=None):
        self.step_count = 0
        self.seed(seed)
        self.cell_population.initialize(h=2**(-5))
        state = 0.0# self.cell_population.init_conditions
        n_cells = self.cell_population.init_conditions.sum()
        # state = float(state) / 1000
        self.previous_cost = state
        self.stacked_states = [state] * self.frame_stack
        return np.array(self.stacked_states, dtype=np.float32), {'n_cells': n_cells}

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]