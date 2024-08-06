from time import time, sleep

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes, ep_reward_infos, ep_training_infos = [], [], {}, {}
        for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
            obs, done, ep_reward, ep_reward_info, t = self.env.reset(), False, 0, {}, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            start_time = time()
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                now = time()
                sleep(max(0, 0.05 - (now - start_time)))
                start_time = now
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_reward_info = {
                    k: ep_reward_info.get(k, 0)
                    + sum(info["reward"][k]) / len(info["reward"][k])
                    for k in info["reward"]
                }
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            for k, v in ep_reward_info.items():
                if k not in ep_reward_infos:
                    ep_reward_infos[k] = []
                ep_reward_infos[k].append(v)
            for k, v in info["training_info"].items():
                if k not in ep_training_infos:
                    ep_training_infos[k] = []
                ep_training_infos[k].append(v)
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_reward_info={k: np.nanmean(v) for k, v in ep_reward_infos.items()},
            episode_training_info={
                k: np.nanmean(v) for k, v in ep_training_infos.items()
            },
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = {k: v.unsqueeze(0).cpu() for k, v in obs.items()}
            obs = TensorDict(obs, batch_size=(1, self.cfg.num_envs), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).repeat(self.cfg.num_envs)
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(
                1,
                self.cfg.num_envs,
            ),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, torch.tensor(True), True
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    tds = torch.cat(self._tds)
                    train_metrics.update(
                        episode_reward=tds["reward"].nansum(0).mean(),
                        episode_success=info["success"].mean(),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self._ep_idx = self.buffer.add(tds)

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = max(
                        1, int(self.cfg.num_envs / self.cfg.steps_per_update)
                    )
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)
