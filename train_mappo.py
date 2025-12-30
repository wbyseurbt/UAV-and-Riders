import argparse
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
import os
import csv
import logging
import warnings
import torch
import multiprocessing

# ==========================================
# 1. 环境设置与日志清理
# ==========================================
os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)

from env import DeliveryUAVEnv

def env_creator(env_config):
    max_steps = env_config.get("max_steps", 200)
    seed = env_config.get("seed", None)
    env = DeliveryUAVEnv(max_steps=max_steps, seed=seed)
    return ParallelPettingZooEnv(env)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if str(agent_id).startswith("rider_"):
        return "rider_policy"
    return "station_policy"

def main():
    # 自动检测 CPU，但在调试阶段我们先手动限制，防止 WSL 内存炸了
    # 建议先设为 4 个 Worker，稳定后再慢慢往上加
    default_workers = 4 

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=default_workers, help="CPU并行采样进程数")
    parser.add_argument("--num-envs-per-worker", type=int, default=5, help="每个进程内的向量化环境数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 检查 GPU
    if torch.cuda.is_available():
        print(f"--- 🚀 显卡已就绪: {torch.cuda.get_device_name(0)} ---")
        num_gpus = 1
    else:
        print("--- ⚠️ 未检测到 GPU，使用 CPU ---")
        num_gpus = 0

    print(f"--- 启动配置: {args.num_workers} Workers x {args.num_envs_per_worker} Envs (总并发: {args.num_workers * args.num_envs_per_worker}) ---")
    
    if ray.is_initialized():
        ray.shutdown()
    
    # 增加 object_store_memory 防止内存溢出报错 (可选，视机器配置而定)
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    register_env("delivery_pz_env", env_creator)

    # --- 获取空间信息 ---
    temp_env_instance = DeliveryUAVEnv(max_steps=args.max_steps, seed=0)
    rider_obs_space = temp_env_instance.observation_spaces["rider_0"]
    rider_act_space = temp_env_instance.action_spaces["rider_0"]
    station_obs_space = temp_env_instance.observation_spaces["station_0"]
    station_act_space = temp_env_instance.action_spaces["station_0"]
    
    policies = {
        "rider_policy": (None, rider_obs_space, rider_act_space, {}),
        "station_policy": (None, station_obs_space, station_act_space, {}),
    }

    # 重新调整 Batch Size
    # 4 workers * 5 envs * 200 steps * 13 agents = 52,000
    # 设为 20000 保证快速响应
    train_batch_size = 20000 

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="delivery_pz_env",
            env_config={"max_steps": args.max_steps, "seed": 0},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            sample_timeout_s=600,
        )
        .resources(
            num_gpus=num_gpus 
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=train_batch_size, 
            minibatch_size=4096,
            # === [修正] 参数改名: num_sgd_iter -> num_epochs ===
            num_epochs=5, 
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["rider_policy", "station_policy"],
        )
    )

    algo = config.build_algo()

    log_filename = "training_wsl_gpu.csv"
    # 创建 checkpoints 目录，防止报错
    os.makedirs("./checkpoints", exist_ok=True)
    print(f"--- 训练开始! 日志文件: {log_filename} ---")
    
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Reward_Mean", "Episode_Len_Mean", "Total_Timesteps", "Steps_Per_Sec"])
        
        for i in range(args.iters):
            result = algo.train()
            
            metrics = result.get("env_runners", {}) or result
            mean_rew = metrics.get("episode_reward_mean", float('nan'))
            mean_len = metrics.get("episode_len_mean", 0)
            total_steps = result.get("num_env_steps_sampled", 0)
            fps = result.get("num_env_steps_sampled_throughput_per_sec", 0)

            print(f"Iter {i+1:03d}/{args.iters} | Reward: {mean_rew:.2f} | FPS: {fps:.0f} | TotalSteps: {total_steps}")
            writer.writerow([i+1, mean_rew, mean_len, total_steps, fps])
            f.flush()

            # [关键修改] 保存模型 Checkpoint
            if (i + 1) % 10 == 0: # 每10轮存一次
                # 指定保存路径，方便查找
                save_dir = os.path.abspath(f"./checkpoints/iter_{i+1:04d}")
                checkpoint_path = algo.save(checkpoint_dir=save_dir)
                print(f"    --> 模型已保存: {checkpoint_path}")

    print("--- 训练结束 ---")
    ray.shutdown()

if __name__ == "__main__":
    main()