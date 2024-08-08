from cell_env import CellEnv
# Use sb3 env checker:
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


#PARAMS OF MODEL:
DT = 0.05
TIMESTEPS = 3000

# Create the environment:
env = CellEnv(dt=DT)
# eval wrapper:
env = TimeLimit(env, TIMESTEPS)
# use the monitor wrapper to log the results:
env = Monitor(env)
eval_env = TimeLimit(CellEnv(dt=DT), TIMESTEPS)
eval_env = Monitor(eval_env)


eval_callback = EvalCallback(eval_env, best_model_save_path=f'./rl-models-dt={DT}/',
                             log_path='./rl-logs/', eval_freq=1000,
                             deterministic=True, render=False,
                             )

model = DQN("MlpPolicy", DummyVecEnv([lambda: Monitor(env)]), verbose=4, tensorboard_log="./rl-logs/",
            exploration_fraction=0.4,
            target_update_interval=5000,
            buffer_size=100000,
            learning_starts=1000,
            learning_rate=0.001,
            device='cuda'
)
model.learn(total_timesteps=1_000_000, tb_log_name="dqn",
            callback=eval_callback)

