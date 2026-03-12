"""Train PPO with TorchVecEnv (GPU-batched). python train.py => torch/cuda/512 envs/1024 steps.

若训练一段时间后显存 OOM：多为保存 checkpoint 时峰值叠加或显存碎片。可尝试：
  - 减少 --n_envs / --batch_size，或增大 --save_every_iters
  - 在运行前设置环境变量减少碎片: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (Linux/Mac: export ...)
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from stable_baselines3.common.callbacks import CallbackList

from scripts.sb3.timing_utils import TimedPPO
from scripts.sb3.gpu_ppo import GpuPPO
from scripts.sb3.savemodel import SaveByIterationCallback
from scripts.sb3.tensorboard import IterationTensorboardCallback
from uavriders.envs.torch_vec_env import TorchVecEnv


def parse_int_list(value: str) -> list[int]:
    s = str(value).strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", "--n-envs", dest="n_envs", type=int, default=1024)
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=1024)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--policy", type=str, default="MlpPolicy",
                        choices=("MlpPolicy", "CnnPolicy", "MultiInputPolicy"))
    parser.add_argument("--device", type=str, default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--n_steps", "--n-steps", dest="n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16384, #65536=2GB
                        help="minibatch size for PPO update (0=full buffer). 32GB 显存建议 16384~32768")
    parser.add_argument("--n_epochs", "--n-epochs", dest="n_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", "--gae-lambda", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", "--clip-range", dest="clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", "--ent-coef", dest="ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", "--vf-coef", dest="vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", "--max-grad-norm", dest="max_grad_norm", type=float, default=0.5)
    parser.add_argument("--net_arch", "--net-arch", dest="net_arch", type=str, default="64,64")
    parser.add_argument("--run_time", "--run-time", dest="run_time", type=str, default="")
    parser.add_argument("--root_logdir", "--root-logdir", dest="root_logdir", type=str, default="./logs")
    parser.add_argument("--save_every_iters", "--save-every-iters", dest="save_every_iters", type=int, default=10)
    parser.add_argument("--no_compile", "--no-compile", dest="no_compile", action="store_true", default=False,
                        help="Disable torch.compile (useful for debugging or if compilation fails)")
    args = parser.parse_args()

    run_time = str(args.run_time).strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_name = str(args.algo).strip() or "ppo"
    run_dir = os.path.abspath(os.path.join(args.root_logdir, algo_name, run_time))
    os.makedirs(run_dir, exist_ok=True)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    torch.set_float32_matmul_precision("high")

    n_envs = int(args.n_envs)
    if n_envs <= 0:
        n_envs = 512

    device = str(args.device).strip() or "cuda"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vec_env = TorchVecEnv(
        num_envs=n_envs,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        device=device,
        compile=not args.no_compile,
    )

    policy_kwargs: dict = {}
    net_arch = parse_int_list(args.net_arch)
    if net_arch:
        policy_kwargs["net_arch"] = net_arch

    n_steps = int(args.n_steps)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        # 0 = 使用整条 rollout 作为一批（显存峰值最高，易 OOM）
        batch_size = max(1, n_steps * n_envs)//4

    model = GpuPPO(
        gpu_env=vec_env,
        policy=str(args.policy),
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

    timesteps_per_iter = n_steps * n_envs
    tb_iter_dir = os.path.join(run_dir, "tb_iter")
    callbacks = [
        IterationTensorboardCallback(tb_dir=tb_iter_dir, timesteps_per_iter=timesteps_per_iter),
        SaveByIterationCallback(
            save_every_iters=int(args.save_every_iters),
            checkpoints_dir=checkpoints_dir,
            timesteps_per_iter=timesteps_per_iter,
        ),
    ]

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
