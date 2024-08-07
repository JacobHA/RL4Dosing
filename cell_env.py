import gymnasium as gym
import numpy as np
import random
from cell_model_pop_fde import Cell_Population

class CellEnv(gym.Env):
    def __init__(self, frame_stack=20, dt=0.15, **kwargs):
        self.dt = dt
        # Use binary actions: apply antibiotic or not
        self.action_space = gym.spaces.Discrete(2)
        # Use continuous observations: cell population (or concentration)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(frame_stack,), dtype=np.float32)
        self.frame_stack = frame_stack
        # Initialize the cell population model
        self.cell_population = Cell_Population(**kwargs)
        # self.previous_cost = None
        self.id = 'CellEnv-v0'


    def step(self, action):
        self.step_count += 1
        # old_cost = self.previous_cost
        # Apply antibiotic if action is 1
        t, tot, cost, p_final = self.cell_population.simulate_population(action, delta_t=self.dt, alpha_mem=0.7)
        # next_state = float(next_state) #/ 1000
        # self.previous_cost = cost
        # todo: make this a deque:
        self.stacked_states.pop(0)
        self.stacked_states.append(cost)
        # Calculate reward
        reward = -cost
        # possible define termination signal if next_state=0 or above some threshold?
        n_cells = tot

        return np.array(self.stacked_states, dtype=np.float32), reward, False, False, {'n_cells': tot}

    def reset(self, seed=None):
        self.step_count = 0
        self.seed(seed)
        self.cell_population.initialize()
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