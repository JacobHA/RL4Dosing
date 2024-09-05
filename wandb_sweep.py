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
    WANDB_DIR = None


DT = 0.01
alpha = 0.8
max_time = int(100 / DT)
env_args = {
    'dt': DT,
    'alpha_mem': alpha,
    'max_timesteps': max_time
}    

def main(log_dir='tf_logs', device='auto'):
    total_timesteps = 300_000
    runs_per_hparam = 1
    avg_auc = 0

    for i in range(runs_per_hparam):
        with wandb.init(sync_tensorboard=True, 
                        # id=unique_id,
                        dir=WANDB_DIR,
                        # sweep_config=sweep_config,
                        # project='iaifi-hackathon',
                        ) as run:  # Ensure sweep_id is specified
            cfg = run.config
            print(run.id)
            config = cfg.as_dict()
            frames = config.pop('frames')
            wandb.log({'frames': frames})
            wandb.log({'terminating': 'True'})
            wandb.log({'truncating as terminate': 'True'})
            wandb.log({'alpha': alpha})
            wandb.log({'dt': DT})

            env = Monitor(CellEnv(frame_stack=frames, **env_args))
            eval_env = Monitor(CellEnv(frame_stack=frames, **env_args))

            eval_callback = EvalCallback(eval_env, best_model_save_path=f'{WANDB_DIR}/sweep-models/{run.name}/',
                             n_eval_episodes=3,
                             log_path='./rl-logs/', eval_freq=10_000,
                             deterministic=True, render=False,
                             )
            # Choose the algo appropriately
            agent = DoubleDQN('MlpPolicy',
                        env,
                        train_freq=(1, "episode"),
                        **config,
                        device=device,
                        tensorboard_log=log_dir)


            agent.learn(total_timesteps=total_timesteps, tb_log_name="dqn",
                        callback=eval_callback)

            # log the sum of eval rewards
            avg_auc += eval_callback.best_mean_reward
            wandb.log({'avg_auc': avg_auc / (i + 1)})

            del agent

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=100)
    args = args.parse_args()
    wandb.login()  # Ensure you are logged in to W&B
    for _ in range(10):
        # Run a hyperparameter sweep with W&B
        print("Running a sweep on W&B...")

        sweep_id = 'jacobhadamczyk/iaifi-hackathon/tysoozhs'  # Ensure this is the correct sweep ID
        wandb.agent(sweep_id, function=main, count=args.count)
        wandb.finish()


