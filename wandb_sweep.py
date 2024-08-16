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


int_hparams = {'train_freq', 'gradient_steps'}

# Load text from settings file
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = None
    

env = CellEnv(dt=0.1, frame_stack=10, alpha_mem=0.7, max_timesteps=500)
eval_env = CellEnv(dt=0.1, frame_stack=10, alpha_mem=0.7, max_timesteps=500)

def main(log_dir='tf_logs', device='auto'):
    total_timesteps = 250_000
    runs_per_hparam = 1
    avg_auc = 0

    for i in range(runs_per_hparam):
        # unique_id = wandb.util.generate_id()[:-1] + f"{i}"
        with wandb.init(sync_tensorboard=True, 
                        # id=unique_id,
                        dir=WANDB_DIR,
                        # sweep_config=sweep_config,
                        # project='iaifi-hackathon',
                        ) as run:  # Ensure sweep_id is specified
            cfg = run.config
            print(run.id)
            config = cfg.as_dict()


            eval_callback = EvalCallback(eval_env, best_model_save_path=f'.sweep-models/{run.name}/',
                             n_eval_episodes=10,
                             log_path='./rl-logs/', eval_freq=5_000,
                             deterministic=True, render=False,
                             )
            # Choose the algo appropriately
            agent = DQN('MlpPolicy',
                        env, **config, device=device, log_interval=5_000,
                        tensorboard_log=log_dir, render=False)



            # Measure the time it takes to learn
            
            agent.learn(total_timesteps=total_timesteps, tb_log_name="dqn",
                        callback=eval_callback)
            del agent

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=100)
    args = args.parse_args()

    # Run a hyperparameter sweep with W&B
    print("Running a sweep on W&B...")
    wandb.login()  # Ensure you are logged in to W&B
    sweep_id = 'jacobhadamczyk/iaifi-hackathon/7v9je8xg'  # Ensure this is the correct sweep ID
    wandb.agent(sweep_id, function=main, count=args.count)
    wandb.finish()
