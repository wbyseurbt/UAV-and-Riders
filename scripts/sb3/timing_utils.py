"""Timing wrapper for PPO (used with TorchVecEnv)."""
import time

import numpy as np
from stable_baselines3 import PPO


class TimedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_count = 0

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.rollout_count += 1
        print(f"\n[Iteration {self.rollout_count}] Starting Rollout Collection...")
        start_time = time.time()
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        total_rollout_time = time.time() - start_time
        print(f"[Timing] Total Data Collection Time: {total_rollout_time:.4f} s")
        return result

    def train(self):
        print(f"[Iteration {self.rollout_count}] Starting Training...")
        start_time = time.time()
        super().train()
        train_time = time.time() - start_time
        print(f"[Timing] Network Training Time: {train_time:.4f} s")
