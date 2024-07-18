from copy import deepcopy

from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from omegaconf import open_dict


class Vectorized:
    """
    Vectorized environment for TD-MPC2 online training.
    """

    def __init__(self, cfg, env_fn):
        super().__init__()
        self.cfg = cfg

        def make_index(i):
            _cfg = deepcopy(cfg)
            _cfg.num_envs = 1
            _cfg.seed = cfg.seed + np.random.randint(1000)
            with open_dict(_cfg) as c:
                c.raisim_config.visualization = i == 0
            return lambda: env_fn(_cfg)

        self.env = AsyncVectorEnv([make_index(i) for i in range(cfg.num_envs)])
        env = make_index(1)()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.max_episode_steps = env.max_episode_steps

    def rand_act(self):
        return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action, render=False):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
