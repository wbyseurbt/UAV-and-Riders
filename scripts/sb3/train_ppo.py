import argparse
import os
from datetime import datetime
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv


class RewardComponentsTensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._sum = defaultdict(float)
        self._count = defaultdict(int)

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
        for k, total in self._sum.items():
            n = self._count.get(k, 0)
            if n <= 0:
                continue
            self.logger.record(f"reward_components/{k}", total / n)
        self._sum.clear()
        self._count.clear()


def make_vec_env(n_envs: int, max_steps: int, seed: int):
    def make_one(rank: int):
        def _init():
            env = DeliveryUAVSingleAgentEnv(max_steps=max_steps, seed=seed + rank)
            return Monitor(env)

        return _init

    return DummyVecEnv([make_one(i) for i in range(n_envs)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", "--n-envs", dest="n_envs", type=int, default=1)
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=200)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--run_time", "--run-time", dest="run_time", type=str, default="")
    parser.add_argument("--root_logdir", "--root-logdir", dest="root_logdir", type=str, default="./logs")
    parser.add_argument("--save_every_iters", "--save-every-iters", dest="save_every_iters", type=int, default=0)
    args = parser.parse_args()

    run_time = str(args.run_time).strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_name = str(args.algo).strip() or "ppo"
    run_dir = os.path.abspath(os.path.join(args.root_logdir, algo_name, run_time))
    os.makedirs(run_dir, exist_ok=True)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    vec_env = make_vec_env(n_envs=int(args.n_envs), max_steps=int(args.max_steps), seed=int(args.seed))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=run_dir,
        seed=int(args.seed),
        device="auto",
    )

    callbacks = [RewardComponentsTensorboardCallback()]

    timesteps_per_iter = int(getattr(model, "n_steps", 2048)) * int(getattr(vec_env, "num_envs", 1))
    for i in range(int(args.iters)):
        model.learn(
            total_timesteps=timesteps_per_iter,
            callback=CallbackList(callbacks),
            tb_log_name=algo_name,
            reset_num_timesteps=False,
        )
        if int(args.save_every_iters) > 0 and (i + 1) % int(args.save_every_iters) == 0:
            iter_path = os.path.join(checkpoints_dir, f"iter_{i+1:04d}.zip")
            model.save(iter_path)
    model.save(final_model_path)

    vec_env.close()


if __name__ == "__main__":
    main()
