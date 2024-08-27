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


DT = 0.01
max_time = int(100 / DT)
env_args = {
    'dt': DT,
    'alpha_mem': 0.7,
    'max_timesteps': max_time
}    

def main(log_dir='tf_logs', device='auto'):
    total_timesteps = 200_000
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
            frames = config.pop('frames')
            wandb.log({'frames': frames})
            wandb.log({'terminating': 'True'})
            wandb.log({'truncating as terminate': 'True'})
            gamma = 0.9995
            wandb.log({'gamma': gamma})
            wandb.log({'dt': DT})

            env = Monitor(CellEnv(frame_stack=frames, **env_args))
            # eval_env = Monitor(
            #     DummyVecEnv([lambda: CellEnv(frame_stack=frames, **env_args)]))
            eval_env = Monitor(CellEnv(frame_stack=frames, **env_args))

            eval_callback = EvalCallback(eval_env, best_model_save_path=f'{WANDB_DIR}/sweep-models/{run.name}/',
                             n_eval_episodes=3,
                             log_path='./rl-logs/', eval_freq=3000,
                             deterministic=True, render=False,
                             )
            # Choose the algo appropriately
            agent = DQN('MlpPolicy',
                        env, 
                        gamma=gamma,
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

    # Run a hyperparameter sweep with W&B
    print("Running a sweep on W&B...")
    wandb.login()  # Ensure you are logged in to W&B
    sweep_id = 'jacobhadamczyk/iaifi-hackathon/7l5dd4d4'  # Ensure this is the correct sweep ID
    wandb.agent(sweep_id, function=main, count=args.count)
    wandb.finish()


