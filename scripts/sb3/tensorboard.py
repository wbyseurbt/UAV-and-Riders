import os
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from torch.utils.tensorboard import SummaryWriter

class IterationTensorboardCallback(BaseCallback):
    def __init__(self, tb_dir: str, timesteps_per_iter: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.tb_dir = str(tb_dir)
        self.timesteps_per_iter = int(timesteps_per_iter)
        self._writer: SummaryWriter | None = None
        self._sum = defaultdict(float)
        self._count = defaultdict(int)

    def _on_training_start(self) -> None:
        os.makedirs(self.tb_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.tb_dir)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True
        for info in infos:
            comps = info.get("reward_components")
            if not isinstance(comps, dict):
                continue
            for k, v in comps.items():
                try:
                    self._sum[str(k)] += float(v)
                    self._count[str(k)] += 1
                except (TypeError, ValueError):
                    continue
        return True

    def _on_rollout_end(self) -> None:
        if self.timesteps_per_iter <= 0:
            return
        iteration = int(getattr(self.model, "num_timesteps", 0)) // self.timesteps_per_iter
        if iteration <= 0:
            return

        for k, total in self._sum.items():
            n = self._count.get(k, 0)
            if n <= 0:
                continue
            value = total / n
            self.logger.record(f"reward_components/{k}", value)
            if self._writer is not None:
                self._writer.add_scalar(f"reward_components/{k}", value, global_step=iteration)

        ep_infos = list(getattr(self.model, "ep_info_buffer", []))
        if ep_infos and self._writer is not None:
            rewards = [float(ep.get("r")) for ep in ep_infos if isinstance(ep, dict) and "r" in ep]
            lengths = [float(ep.get("l")) for ep in ep_infos if isinstance(ep, dict) and "l" in ep]
            if rewards:
                self._writer.add_scalar("rollout/ep_rew_mean", float(np.mean(rewards)), global_step=iteration)
            if lengths:
                self._writer.add_scalar("rollout/ep_len_mean", float(np.mean(lengths)), global_step=iteration)

        if self._writer is not None:
            self._writer.flush()

        self._sum.clear()
        self._count.clear()

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
