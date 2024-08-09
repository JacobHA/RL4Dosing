import wandb
import os
import gymnasium
import argparse
import yaml
import sys
from gymnasium.wrappers import TimeLimit

sys.path.append('avg_rwds')
from avg_rwds.UAgent import UAgent

int_hparams = {'train_freq', 'gradient_steps'}

# Load text from settings file
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = None
    
from cell_env import CellEnv

env = CellEnv(dt=0.15, frame_stack=10, alpha_mem=0.7)
env = TimeLimit(env, 1000)

def main(log_dir='tf_logs', device='auto'):
    total_timesteps = 250_000
    runs_per_hparam = 2
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


            # Choose the algo appropriately
            agent = UAgent(env, **config, device=device, log_interval=5000,
                           tensorboard_log=log_dir, render=False)

            # Measure the time it takes to learn
            agent.learn(total_timesteps=total_timesteps)
            avg_auc += agent.eval_auc
            wandb.log({'avg_auc': avg_auc / runs_per_hparam})
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
