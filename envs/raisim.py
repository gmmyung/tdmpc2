import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import OmegaConf, open_dict
from envs.wrappers.time_limit import TimeLimit
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
import torch

# TODO: Remove [0] indexing by directly using RaisimGymEnv
class RaisimEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.metadata["render_fps"] = int(1.0 / self.cfg.raisim_config.control_dt)
        self.env = VecEnv(
            RaisimGymEnv(cfg.raisim_resource_path, OmegaConf.to_yaml(cfg.raisim_config))
        )
        self.env.seed(cfg.raisim_config.seed)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.env.num_acts,), dtype=np.float32
        )
        if cfg.raisim_config.visualize:
            self.env.turn_on_visualization()

    def reset(self, **kwargs):
        self.env.reset()
        obs = self.env.observe().astype(np.float32)
        self.env.curriculum_callback()
        return obs, {}

    def step(self, action):
        # rewards, dones = self.env.step(action)
        rewards, dones = self.env.step(
                action
        )
        reward_infos = self.env.get_reward_info()
        training_infos = self.env.get_training_info()
        obs = np.stack(self.env.observe()).astype(np.float32)
        return (
            obs,
            rewards,
            False,
            False,
            {
                "success": ~dones,
                "reward": reward_infos,
                "training_info": training_infos,
            },
        )
    
    def rand_act(self):
        return np.random.uniform(-1, 1, (self.env.num_envs, self.env.num_acts)).astype(np.float32)

    def render(self, mode="depth"):
        if mode == "depth":
            im = (
                np.nan_to_num(np.stack(self.env.depth_image()), nan=20)
                .clip(0, 20)
                .transpose(0, 2, 1)
            )
            assert isinstance(im, np.ndarray)
            return np.expand_dims(im, 1)

    def close(self):
        if self.cfg.raisim_config.visualization:
            self.env.turn_off_visualization()
        self.env.close()


def make_env(cfg):
    if cfg.task != "raisim":
        raise ValueError("Unknown task:", cfg.task)
    env = RaisimEnv(cfg)
    env = TimeLimit(
        env,
        max_episode_steps=cfg.raisim_config.max_time // cfg.raisim_config.control_dt,
    )
    env.max_episode_steps = env._max_episode_steps
    return env
