from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch


class MultiModalWrapper(gym.Wrapper):
    """
    Wrapper for multi-modal environments.
    """

    def __init__(self, cfg, env, num_frames=8, render_size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(num_frames, render_size, render_size),
                    dtype=np.float32,
                ),
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_frames * env.observation_space.shape[0],))
            }
        )
        self._frames = deque([], maxlen=num_frames)
        self._state_frames = deque([], maxlen=num_frames)
        self._render_size = render_size

    def _get_images(self):
        frame = self.env.render()
        self._frames.append(frame)
        return torch.from_numpy(np.concatenate(self._frames, axis=-3))

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._frames.maxlen):
            self._get_images()
            self._state_frames.append(obs)
        return {"rgb": self._get_images(), "state": torch.from_numpy(np.concatenate(self._state_frames, axis=-1))}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._state_frames.append(obs)
        return (
            {"rgb": self._get_images(), "state": torch.from_numpy(np.concatenate(self._state_frames, axis=-1))},
            reward,
            done,
            info,
        )
