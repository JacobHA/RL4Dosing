from cell_env import CellEnv
# Use sb3 env checker:
from stable_baselines3.common.env_checker import check_env
from ddqn import DoubleDQN

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

def main(alpha, frames, GAMMA):
    env_args = {
        "max_timesteps": 10000,
        "alpha_mem": alpha,
        "dt": 0.01,
        "frame_stack": frames,
    }
    env = CellEnv(**env_args)
    # use the monitor wrapper to log the results:
    env = Monitor(env)
    eval_env = CellEnv(**env_args)
    eval_env = Monitor(eval_env)


    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./rl-models-sde_{alpha}/',
                                n_eval_episodes=1,
                                log_path='./rl-logs/', eval_freq=10_000,
                                deterministic=True, render=False,
                                )


    model = DoubleDQN("MlpPolicy", env,
                        verbose=4, tensorboard_log=f"./rl-logs/{alpha}_{frames}_{GAMMA}/",
                        exploration_fraction=0.2,
                        exploration_final_eps=0.0,
                        target_update_interval=1000,
                        buffer_size=100_000,
                        gradient_steps=-1,
                        train_freq=(1, "episode"),
                        learning_starts=0,
                        learning_rate=0.00036,
                        batch_size=32,
                        gamma=GAMMA,
    )
    model.learn(total_timesteps=100_000, tb_log_name="dqn",
                callback=eval_callback)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.995)
    args = parser.parse_args()
    main(args.alpha, args.frames, args.gamma)
