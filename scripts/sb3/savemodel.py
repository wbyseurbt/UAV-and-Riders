import gc
import os

import torch
from stable_baselines3.common.callbacks import BaseCallback


class SaveByIterationCallback(BaseCallback):
    def __init__(self, save_every_iters: int, checkpoints_dir: str, timesteps_per_iter: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_every_iters = int(save_every_iters)
        self.checkpoints_dir = str(checkpoints_dir)
        self.timesteps_per_iter = int(timesteps_per_iter)
        self._last_saved_iter = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.save_every_iters <= 0:
            return
        if self.timesteps_per_iter <= 0:
            return
        current_iter = int(getattr(self.model, "num_timesteps", 0)) // self.timesteps_per_iter
        if current_iter <= 0:
            return
        if current_iter % self.save_every_iters != 0:
            return
        if current_iter == self._last_saved_iter:
            return
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        iter_path = os.path.join(self.checkpoints_dir, f"iter_{current_iter:04d}.zip")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        try:
            self.model.save(iter_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                if self.verbose >= 1:
                    print(f"[SaveByIteration] OOM 保存 checkpoint 失败 (iter {current_iter})，可增大 --save_every_iters 或减小 --batch_size")
                raise
            raise
        self._last_saved_iter = current_iter
