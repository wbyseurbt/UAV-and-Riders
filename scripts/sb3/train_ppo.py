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


from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv

from scripts.sb3.savemodel import SaveByIterationCallback
from scripts.sb3.tensorboard import IterationTensorboardCallback


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

    # Prepare run directory, saved models, and tensorboard log directory
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
