import time
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from typing import Optional, Union, List, Tuple, Dict, Any, get_type_hints
from typeguard import typechecked
import wandb
from avg_rwds.utils import env_id_to_envs, is_tabular, log_class_vars

HPARAM_ATTRS = {
    'beta': 'beta',
    'learning_rate': 'learning_rate',
    'batch_size': 'batch_size',
    'buffer_size': 'buffer_size',
    'target_update_interval': 'target_update_interval',
    'tau': 'tau',
    'theta_update_interval': 'theta_update_interval',
    'hidden_dim': 'hidden_dim',
    'num_nets': 'num_nets',
    'tau_theta': 'tau_theta',
    'gradient_steps': 'gradient_steps',
    'train_freq': 'train_freq',
    'max_grad_norm': 'max_grad_norm',
    'learning_starts': 'learning_starts',
}

LOG_PARAMS = {
    'time/env. steps': 'env_steps',
    'eval/avg_reward': 'avg_eval_rwd',
    'eval/auc': 'eval_auc',
    'time/num. episodes': 'num_episodes',
    'time/fps': 'fps',
    'time/num. updates': '_n_updates',
    'rollout/beta': 'beta',
    'train/lr': 'lr',
}

int_args = ['batch_size',
            'buffer_size',
            'target_update_interval',
            'theta_update_interval',
            'hidden_dim',
            'num_nets',
            'gradient_steps',
            'train_freq',
            'max_grad_norm',
            'learning_starts']


str_to_aggregator = {'min': lambda x, dim: torch.min(x, dim=dim)[0],
                     'max': lambda x, dim: torch.max(x, dim=dim)[0],
                     'mean': lambda x, dim: (torch.mean(x, dim=dim))}

# use get_type_hints to throw errors if the user passes in an invalid type:


class BaseAgent:
    @typechecked
    def __init__(self,
                 env_id: Union[str, gym.Env],
                 learning_rate: float = 1e-3,
                 beta: float = 0.1,
                 beta_schedule: str = 'none',
                 batch_size: int = 64,
                 buffer_size: int = 100_000,
                 target_update_interval: int = 10_000,
                 tau: float = 1.0,
                 theta_update_interval: int = 1,
                 hidden_dim: int = 64,
                 num_nets: int = 2,
                 tau_theta: float = 0.995,
                 gradient_steps: int = 1,
                 train_freq: Union[int, Tuple[int, str]] = 1,
                 max_grad_norm: float = 10,
                 learning_starts=5_000,
                 aggregator: str = 'max',
                 # torch.nn.functional.HuberLoss(),
                 loss_fn: torch.nn.modules.loss = torch.nn.functional.mse_loss,
                 device: Union[torch.device, str] = "auto",
                 render: bool = False,
                 tensorboard_log: Optional[str] = None,
                 log_interval: int = 1_000,
                 save_checkpoints: bool = False,
                 use_wandb: bool = False,
                 scheduler_str: str = 'none',
                 beta_end: Optional[float] = None,
                 seed: Optional[int] = None,
                 ) -> None:


        self.env, self.eval_env = env_id_to_envs(env_id, render)

        if hasattr(self.env.unwrapped.spec, 'id'):
            self.env_str = self.env.unwrapped.spec.id
        elif hasattr(self.env.unwrapped, 'id'):
            self.env_str = self.env.unwrapped.id
        else:
            self.env_str = str(self.env.unwrapped)

        self.is_tabular = is_tabular(self.env)

        self.learning_rate = learning_rate
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.theta_update_interval = theta_update_interval
        self.train_freq = train_freq
        if isinstance(train_freq, tuple):
            raise NotImplementedError("train_freq as a tuple is not supported yet.\
                                       \nEnter int corresponding to env_steps")
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None
        self.learning_starts = learning_starts
        self.use_wandb = use_wandb
        self.aggregator = aggregator
        self.tensorboard_log = tensorboard_log
        self.aggregator_fn = str_to_aggregator[aggregator]
        self.avg_eval_rwd = None
        self.fps = None
        self.beta_end = beta_end
        self.scheduler_str = scheduler_str
        self.train_this_step = False
        # Track the rewards over time:
        self.step_to_avg_eval_rwd = {}

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=True,
                                          device=device)
        # assert isinstance(self.env.action_space, gym.spaces.Discrete), \
        #     "Only discrete action spaces are supported."
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.nA = self.env.action_space.n

        self.theta = torch.Tensor([0.0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        self._n_updates = 0
        self.env_steps = 0
        # self._initialize_networks()
        self.loss_fn = loss_fn

    def log_hparams(self, logger):
        # Log the hparams:
        log_class_vars(self, logger, HPARAM_ATTRS)
        logger.dump()

    def _initialize_networks(self):
        raise NotImplementedError

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        """
        Sample an action from the policy of choice
        """
        raise NotImplementedError

    def gradient_descent(self, batch, grad_step: int):
        """
        Do a gradient descent step
        """
        raise NotImplementedError

    def _train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        self.new_theta_pending = 0
        self.new_theta_counter = 1
        # Increase update counter
        self._n_updates += gradient_steps
        # average self.theta over multiple gradient steps
        self.new_thetas = torch.zeros(gradient_steps).to(self.device)
        for grad_step in range(gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.replay_buffer.sample(batch_size)

            loss = self.gradient_descent(batch, grad_step)
            self.optimizers.zero_grad()

            # Clip gradient norm
            loss.backward()
            self.model.clip_grad_norm(self.max_grad_norm)
            self.optimizers.step()

        self.new_thetas = torch.clamp(self.new_thetas, min=-50, max=50)
        new_theta = self.new_thetas.mean(dim=0)
        self.logger.record(f"train/new_theta", new_theta.item())

        # Can't use env_steps b/c we are inside the learn function which is called only
        # every train_freq steps:
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + \
                (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps: int, early_stop: dict = {}) -> bool:
        """
        Train the agent for total_timesteps
        """
        stop_steps = early_stop.get('steps', 0)
        if stop_steps > 0:
            assert stop_steps % self.log_interval == 0, \
                "early_stop['steps'] must be a multiple of log_interval, or will never be checked"
        stop_reward = early_stop.get('reward', -np.inf)
        self.betas = self._beta_scheduler(self.beta_schedule, total_timesteps)

        # Start a timer to log fps:
        self.initial_time = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action_freqs = torch.zeros(self.nA)

            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            avg_ep_len = 0
            entropy = 0
            while not done and self.env_steps < total_timesteps:
                action, kl = self.exploration_policy(state)
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs[action] += 1
                # Add KL divergence bw the current policy and the prior:
                entropy += float(kl)

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                self._on_step()
                avg_ep_len += 1
                done = terminated or truncated
                self.rollout_reward += reward

                self.train_this_step = (self.train_freq == -1 and terminated) or \
                    (self.train_freq != -1 and self.env_steps %
                     self.train_freq == 0)

                # Add the transition to the replay buffer:
                sarsa = (state, next_state, action, reward, terminated)
                self.replay_buffer.add(*sarsa, [infos])
                state = next_state
                if self.env_steps % self.log_interval == 0:
                    self._log_stats()
                    # if (self.env_steps > stop_steps):
                    # this was too strict. trying this:
                    if self.env_steps == stop_steps:
                        if (self.avg_eval_rwd < stop_reward):
                            wandb.log({'early_stop': True})
                            return True

            if terminated:
                # self.rollout_reward += 0
                avg_ep_len += 1
            if done:
                self.rollout_reward
                self.logger.record("rollout/ep_reward", self.rollout_reward)
                free_energy = (self.rollout_reward + 1/self.beta * entropy)
                try:
                    free_energy = free_energy.item()
                except:
                    pass
                # entropy = 0
                self.logger.record("rollout/neg_free_energy",
                                   free_energy / avg_ep_len)
                self.logger.record("rollout/avg_entropy", entropy / avg_ep_len)
                self.logger.record("rollout/avg_episode_length", avg_ep_len)
                self.logger.record("rollout/avg_reward_rate",
                                   self.rollout_reward / avg_ep_len)
                if self.use_wandb:
                    wandb.log({'rollout/reward': self.rollout_reward})
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs /= action_freqs.sum()
                    for i, freq in enumerate(action_freqs):
                        # As percentage:
                        self.logger.record(
                            f'rollout/action {i} (%)', freq.item() * 100)

        return False

    def _on_step(self):
        """
        This method is called after every step in the environment
        """
        self.beta = self.betas[self.env_steps]
        self.env_steps += 1

        if self.train_this_step:
            if self.env_steps > self.learning_starts:
                self._train(self.gradient_steps, self.batch_size)

        if self.env_steps % self.target_update_interval == 0:
            self._update_target()

        if self.env_steps % self.log_interval == 0:
            # Log info from this training cycle:
            self.logger.record("train/theta", self.theta.item())

    def _log_stats(self):
        # end timer:
        t_final = time.thread_time_ns()
        # fps averaged over log_interval steps:
        self.fps = self.log_interval / \
            ((t_final - self.initial_time + 1e-16) / 1e9)

        if self.env_steps > 0:
            self.avg_eval_rwd = self.evaluate()
            self.eval_auc += self.avg_eval_rwd
        if self.save_checkpoints:
            raise NotImplementedError
            torch.save(self.online_logu.state_dict(),
                       'sql-policy.para')
        # Get the current learning rate from the optimizer:
        self.lr = self.optimizers.get_lr()
        log_class_vars(self, self.logger, LOG_PARAMS, use_wandb=self.use_wandb)

        if self.use_wandb:
            wandb.log({'env_steps': self.env_steps,
                       'eval/avg_reward': self.avg_eval_rwd})
        self.logger.dump(step=self.env_steps)
        self.initial_time = time.thread_time_ns()

    def evaluate(self, n_episodes=10) -> float:
        # run the current policy and return the average reward
        self.initial_time = time.process_time_ns()
        avg_reward = 0.
        # log the action frequencies:
        action_freqs = torch.zeros(self.nA)
        n_steps = 0
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.evaluation_policy(state)
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action_freqs[action] += 1
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # log the action frequencies:
            action_freqs /= n_episodes
            action_freqs /= action_freqs.sum()
            for i, freq in enumerate(action_freqs):
                # As percentage:
                self.logger.record(f'eval/action {i} (%)', freq.item() * 100)
        self.logger.record('eval/avg_episode_length', n_steps / n_episodes)
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time + 1e-12) / 1e9
        eval_fps = n_steps / eval_time
        self.logger.record('eval/time', eval_time)
        self.logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        self.step_to_avg_eval_rwd[self.env_steps] = avg_reward
        return avg_reward

    def _beta_scheduler(self, beta_schedule, total_timesteps):
        # setup beta scheduling
        if beta_schedule == 'exp':
            self.betas = torch.exp(torch.linspace(np.log(self.beta), np.log(
                self.beta_end), total_timesteps)).to(self.device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(
                self.beta, self.beta_end, total_timesteps).to(self.device)
        elif beta_schedule == 'none':
            self.betas = torch.tensor(
                [self.beta] * total_timesteps).to(self.device)
        else:
            raise NotImplementedError(
                f"Unknown beta schedule: {beta_schedule}")
        return self.betas