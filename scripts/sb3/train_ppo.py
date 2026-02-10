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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv


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
        self.model.save(iter_path)
        self._last_saved_iter = current_iter


def make_vec_env(n_envs: int, max_steps: int, seed: int, vec_env_type: str, start_method: str):
    def make_one(rank: int):
        def _init():
            env = DeliveryUAVSingleAgentEnv(max_steps=max_steps, seed=seed + rank)
            return Monitor(env)

        return _init

    env_fns = [make_one(i) for i in range(n_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)


def parse_int_list(value: str) -> list[int]:
    s = str(value).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out: list[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", "--n-envs", dest="n_envs", type=int, default=0)
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=1024)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument(
        "--vec_env",
        "--vec-env",
        dest="vec_env",
        type=str,
        default="subproc",
        choices=("dummy", "subproc"),
    )
    parser.add_argument(
        "--start_method",
        "--start-method",
        dest="start_method",
        type=str,
        default="forkserver",
        choices=("forkserver", "spawn", "fork"),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        choices=("MlpPolicy", "CnnPolicy", "MultiInputPolicy"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--n_steps", "--n-steps", dest="n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=0)
    parser.add_argument("--n_epochs", "--n-epochs", dest="n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", "--gae-lambda", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", "--clip-range", dest="clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", "--ent-coef", dest="ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", "--vf-coef", dest="vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", "--max-grad-norm", dest="max_grad_norm", type=float, default=0.5)
    parser.add_argument("--net_arch", "--net-arch", dest="net_arch", type=str, default="")
    parser.add_argument("--run_time", "--run-time", dest="run_time", type=str, default="")
    parser.add_argument("--root_logdir", "--root-logdir", dest="root_logdir", type=str, default="./logs")
    parser.add_argument("--save_every_iters", "--save-every-iters", dest="save_every_iters", type=int, default=10)
    args = parser.parse_args()

    run_time = str(args.run_time).strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_name = str(args.algo).strip() or "ppo"
    run_dir = os.path.abspath(os.path.join(args.root_logdir, algo_name, run_time))
    os.makedirs(run_dir, exist_ok=True)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    requested_n_envs = int(args.n_envs)
    if requested_n_envs <= 0:
        requested_n_envs = max(1, int(os.cpu_count() or 1))

    vec_env = make_vec_env(
        n_envs=requested_n_envs,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        vec_env_type=str(args.vec_env),
        start_method=str(args.start_method),
    )

    policy = str(args.policy).strip() or "MlpPolicy"
    device = str(args.device).strip() or "auto"
    if device == "auto" and policy == "MlpPolicy":
        device = "cpu"

    policy_kwargs: dict = {}
    net_arch = parse_int_list(args.net_arch)
    if net_arch:
        policy_kwargs["net_arch"] = net_arch

    n_steps = int(args.n_steps)
    num_envs = int(getattr(vec_env, "num_envs", 1))
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = max(1, n_steps * num_envs)

    model = PPO(
        policy=policy,
        env=vec_env,
        verbose=1,
        seed=int(args.seed),
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
    )

    timesteps_per_iter = int(getattr(model, "n_steps", 2048)) * int(getattr(vec_env, "num_envs", 1))
    tb_iter_dir = os.path.join(run_dir, "tb_iter")
    callbacks = [IterationTensorboardCallback(tb_dir=tb_iter_dir, timesteps_per_iter=timesteps_per_iter)]

    callbacks.append(
        SaveByIterationCallback(
            save_every_iters=int(args.save_every_iters),
            checkpoints_dir=checkpoints_dir,
            timesteps_per_iter=timesteps_per_iter,
        )
    )

    model.learn(
        total_timesteps=timesteps_per_iter * int(args.iters),
        callback=CallbackList(callbacks),
        tb_log_name=algo_name,
        reset_num_timesteps=True,
    )
    model.save(final_model_path)

    vec_env.close()


if __name__ == "__main__":
    main()
