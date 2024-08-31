from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch


class MultiModalWrapper(gym.Wrapper):
    """
    Wrapper for multi-modal environments.
    """

    def __init__(self, cfg, env, num_frames=2):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        sample_imgs = self.env.render()
        observation_space = {}
        observation_space["state"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_frames * env.observation_space.shape[0],))
        for i in range(len(sample_imgs)):
            observation_space["rgb" + str(i)] = gym.spaces.Box(low=0, high=255, shape=(sample_imgs[0].shape[-3] * num_frames,) + sample_imgs[0].shape[-2:], dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(observation_space)
        self._frames = [deque([], maxlen=num_frames) for _ in range(len(sample_imgs))]
        self._state_frames = deque([], maxlen=num_frames)

    def _get_images(self):
        images = self.env.render()
        # self._frames.append(frame)
        for i in range(len(images)):
            self._frames[i].append(images[i])
        return [torch.from_numpy(np.concatenate(f, axis=-3)) for f in self._frames]

    def reset(self):
        state = self.env.reset()
        for _ in range(self._state_frames.maxlen):
            self._state_frames.append(state)
            self._get_images()
        obs = {}
        obs['state'] = torch.from_numpy(np.concatenate(self._state_frames, axis=-1))
        for i in range(len(self._frames)):
            obs["rgb" + str(i)] = self._get_images()[i]
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._state_frames.append(state)
        obs = {}
        obs['state'] = torch.from_numpy(np.concatenate(self._state_frames, axis=-1))
        for i in range(len(self._frames)):
            obs["rgb" + str(i)] = self._get_images()[i]
        return (
            obs,
            reward,
            done,
            info,
        )
