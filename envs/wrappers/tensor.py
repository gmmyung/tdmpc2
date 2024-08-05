from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        return torch.from_numpy(self.env.rand_act())

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None):
        return self._obs_to_tensor(self.env.reset()[0])

    def step(self, action):
        obs, reward, done, info = self.env.step(action.numpy())
        info["success"] = info["success"].astype(np.float32)
        return (
            self._obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            done,
            info,
        )
