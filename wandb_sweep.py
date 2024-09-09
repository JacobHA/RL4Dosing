import wandb
import os
import gymnasium
import argparse
import yaml
import sys
from gymnasium.wrappers import TimeLimit
from cell_env import CellEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.dqn import DQN
from ddqn import DoubleDQN

int_hparams = {'train_freq', 'gradient_steps'}

# Load text from settings file
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = 'wandb_logs'


DT = 0.01
alpha = 0.8
max_time = int(100 / DT)


def main(device='auto'):
    total_timesteps = 300_000
    avg_auc = 0

    with wandb.init(sync_tensorboard=True, 
                    dir=WANDB_DIR,
                    ) as run:
        cfg = run.config
        config = cfg.as_dict()
        frames = config.pop('frames')
        wandb.log({'frames': frames})
        wandb.log({'terminating': 'True'})
        wandb.log({'truncating as terminate': 'True'})
        wandb.log({'alpha': alpha})
        wandb.log({'dt': DT})
        # check if dt is in config
        if 'dt' in config:
            dt = config['dt']
            max_time = int(100 / dt)
            # reassign the max_time
            env_args = {
                "max_timesteps": max_time,
                "dt": dt,
            }

        env = Monitor(CellEnv(frame_stack=frames, **env_args))
        eval_env = Monitor(CellEnv(frame_stack=frames, **env_args))

        eval_callback = EvalCallback(eval_env, best_model_save_path=f'{WANDB_DIR}/sweep-models/{run.name}/',
                            n_eval_episodes=3,
                            log_path='./rl-logs/', eval_freq=10_000,
                            deterministic=True, render=False,
                            )

        agent = DoubleDQN('MlpPolicy',
                    env,
                    train_freq=(1, "episode"),
                    **config,
                    device=device,
                    tensorboard_log=WANDB_DIR)


        agent.learn(total_timesteps=total_timesteps, tb_log_name="dqn",
                    callback=eval_callback)

        best_eval = eval_callback.best_mean_reward
        wandb.log({'best_eval': best_eval})

        del agent

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=100)
    args = args.parse_args()
    wandb.login()  # Ensure you are logged in to W&B
    for _ in range(10):
        # Run a hyperparameter sweep with W&B
        print("Running a sweep on W&B...")
        USERNAME = os.getenv("WANDB_USERNAME", "enter_username_here")
        PROJECT_NAME = os.getenv("WANDB_PROJECT", "enter_project_name_here")
        SWEEP_ID = os.getenv("WANDB_SWEEP_ID", "enter_sweep_id_here")
        sweep_id = f'{USERNAME}/{PROJECT_NAME}/{SWEEP_ID}'  
        # Ensure this is the correct sweep ID by accessing
        # https://wandb.ai/{USERNAME}/{PROJECT_NAME}/sweeps/{SWEEP_ID}
        wandb.agent(sweep_id, function=main, count=args.count)
        wandb.finish()


