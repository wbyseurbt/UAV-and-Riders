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
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--run-time", type=str, default="")
    parser.add_argument("--root-logdir", type=str, default="./logs")
    parser.add_argument("--save-every-steps", type=int, default=0)
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
    if int(args.save_every_steps) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(args.save_every_steps),
                save_path=checkpoints_dir,
                name_prefix="model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    model.learn(total_timesteps=int(args.timesteps), callback=CallbackList(callbacks), tb_log_name=algo_name)
    model.save(final_model_path)

    vec_env.close()


if __name__ == "__main__":
    main()
