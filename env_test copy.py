from cell_env import CellEnv
# Use sb3 env checker:
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
env = CellEnv()
# use the monitor wrapper to log the results:
env = Monitor(env)
eval_env = CellEnv()
eval_env = Monitor(eval_env)

env.reset()
for _ in range(1000):
    print(env.step(0))
env.step(0)