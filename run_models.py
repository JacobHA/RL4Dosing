import numpy as np
from plot_utils import evaluate_model
import os
WANDB_DIR = os.getenv("WANDB_DIR", ".sweep-models")
if "." not in WANDB_DIR:
    WANDB_DIR += "/sweep-models"

# Example usage:
env_args = {
    "max_timesteps": 10000,
    "dt": 0.01,
}


## REDO 1.0 sweep on other values, check hparams on 0.7 and run on  DDQN!

alpha_to_model = {
    0.6: 'cerulean-sweep-100', #18.6
    0.7: 'rich-sweep-67', 
    # 0.7: 'winter-sweep-36', #24.2
    0.8: 'apricot-sweep-139', #35.3
    0.9: 'rare-sweep-383', #54.1
    1.0: 'brisk-sweep-198', #88.5
}

alpha_to_frames = {
    0.6: 40,
    0.7: 5,
    0.8: 50,
    0.9: 2,
    1.0: 5,
}

def run_alpha_eval(alpha):
    assert alpha in alpha_to_model.keys(), f"Alpha value {alpha} not found in swept values dict."

    env_args['frame_stack'] = alpha_to_frames[alpha]
    env_args['alpha_mem'] = alpha

    model_str = f"{WANDB_DIR}/{alpha_to_model[env_args['alpha_mem']]}/best_model.zip"

    print(f"Evaluating model {model_str} for alpha={alpha}...")
    all_obs, all_actions, all_fractions = evaluate_model(env_args, 1, model_str, multiprocess=False)

    # Save to npy file
    alpha_data = np.vstack([all_obs, all_actions, all_fractions])
    np.save(f"alpha={alpha}_data.npy", alpha_data)


import multiprocessing as mp
# Do all alpha values in parallel
MAX_CPUS = mp.cpu_count() - 1
num_cpus = min(MAX_CPUS, len(alpha_to_model))

run_alphas = [0.7]
with mp.Pool(num_cpus) as pool:
    pool.map(run_alpha_eval, run_alphas)

