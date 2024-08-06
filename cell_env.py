import gymnasium as gym
import numpy as np
import random
from cell_model_pop_fde import Cell_Population

class CellEnv(gym.Env):
    def __init__(self, **kwargs):
        # Use binary actions: apply antibiotic or not
        self.action_space = gym.spaces.Discrete(2)
        # Use continuous observations: cell population (or concentration)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Initialize the cell population model
        self.cell_population = Cell_Population(**kwargs)


    def step(self, action):
        self.step_count += 1
        # Apply antibiotic if action is 1
        t, tot, cost, next_state = self.cell_population.simulate_population(action, delta_t=5)
        next_state = float(next_state) / 1000
        # Calculate reward
        reward = -cost
        # possible define termination signal if next_state=0 or above some threshold?

        return np.array([next_state], dtype=np.float32), reward, False, False, {}

    def reset(self, seed=None):
        self.step_count = 0
        self.seed(seed)
        self.cell_population.initialize()
        state, _ = self.cell_population.init_conditions
        state = float(state) / 1000

        return np.array([state], dtype=np.float32), {}

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]