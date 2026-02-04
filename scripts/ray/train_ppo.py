import argparse
import csv
import logging
import multiprocessing
import os
import warnings
from datetime import datetime
import sys
from pathlib import Path

import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.append(str(Path(__file__).resolve().parents[2]))

from uavriders.rl.rllib_integration import make_single_gym_env


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", type=str, default="ray_ppo")
    parser.add_argument("--run-time", type=str, default="")
    parser.add_argument("--root-logdir", type=str, default="./logs")
    parser.add_argument("--save-every-iters", type=int, default=10)
    args = parser.parse_args()

    run_time = str(args.run_time).strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_name = str(args.algo).strip() or "ray_ppo"
    run_dir = os.path.abspath(os.path.join(args.root_logdir, algo_name, run_time))
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    total_cores = 4
    num_workers = max(1, total_cores - 2)
    num_envs_per_worker = 5
    total_concurrency = num_workers * num_envs_per_worker

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        num_gpus = 1
        print(f"--- 🚀 显卡火力全开: {gpu_name} ---")
    else:
        num_gpus = 0
        print("--- ⚠️ 未检测到 GPU，使用 CPU 训练 ---")

    print(f"--- 🔥 CPU 性能拉满模式 ---")
    print(f"    检测到总核心数: {total_cores}")
    print(f"    分配 Workers: {num_workers}")
    print(f"    单 Worker 环境数: {num_envs_per_worker}")
    print(f"    总并发采样环境: {total_concurrency}")
    print("-----------------------------")

    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=total_cores, ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)

    register_env("delivery_single_env", make_single_gym_env)

    train_batch_size = 120000
    sgd_minibatch_size = 8192

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            env="delivery_single_env",
            env_config={"max_steps": args.max_steps, "seed": args.seed},
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            batch_mode="truncate_episodes",
            rollout_fragment_length="auto",
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            sample_timeout_s=600,
            num_cpus_per_env_runner=1,
        )
        .resources(num_gpus=num_gpus)
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=train_batch_size,
            minibatch_size=sgd_minibatch_size,
            num_epochs=5,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
            model={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        )
    )

    algo = config.build_algo()

    log_filename = os.path.join(run_dir, "training_log_optimized.csv")

    with open(log_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Reward_Mean", "Episode_Len_Mean", "Total_Timesteps", "Steps_Per_Sec"])

        print(f"--- 训练开始! 日志: {log_filename} ---")

        for i in range(args.iters):
            result = algo.train()

            metrics = result.get("env_runners", {}) or result
            mean_rew = metrics.get("episode_reward_mean", float("nan"))
            mean_len = metrics.get("episode_len_mean", 0)
            total_steps = result.get("num_env_steps_sampled", 0)
            fps = result.get("num_env_steps_sampled_throughput_per_sec", 0)

            timers = result.get("timers", {})
            _ = timers.get("env_runner_sampling_timer", 0)
            _ = timers.get("learner_grad_update_timer", 0)
            _ = timers.get("synch_weights_timer", 0)

            print(f"Iter {i+1:03d} | FPS: {fps:.0f} | Rew: {mean_rew:.2f}")
            print(f"    🔍 原始计时数据 (Debug): {timers}")

            writer.writerow([i + 1, mean_rew, mean_len, total_steps, fps])
            f.flush()

            if int(args.save_every_iters) > 0 and (i + 1) % int(args.save_every_iters) == 0:
                save_dir = os.path.abspath(os.path.join(checkpoints_dir, f"iter_{i+1:04d}"))
                algo.save(checkpoint_dir=save_dir)
                print("    --> 模型已保存")

    print("--- 训练结束 ---")
    ray.shutdown()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
