import argparse
import os
import csv
import logging
import warnings
import multiprocessing
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

# 引入你的环境
from env import DeliveryUAVEnv

# ==========================================
# 0. 全局配置与清理
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
# 让 Ray 相信我们是在容器里也要用多核
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1" 

warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)

# ==========================================
# 1. 环境注册函数
# ==========================================
def env_creator(env_config):
    max_steps = env_config.get("max_steps", 200)
    seed = env_config.get("seed", None)
    # 确保你的环境类支持 seed 参数，如果不支持请去掉
    env = DeliveryUAVEnv(max_steps=max_steps, seed=seed)
    return ParallelPettingZooEnv(env)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if str(agent_id).startswith("rider_"):
        return "rider_policy"
    return "station_policy"

# ==========================================
# 2. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- 🤖 硬件自动检测 (关键优化) ---
    # 获取物理 CPU 核心数
    total_cores = 25
    num_workers = max(1, total_cores - 2)
    
    # 每个 Worker 并行跑的环境数。
    # 如果环境计算量小，可以设大一点 (5-10)。如果环境很重，设小一点 (1-2)。
    num_envs_per_worker = 5
    
    # 计算总并发数 (用于检查)
    total_concurrency = num_workers * num_envs_per_worker

    # 检查 GPU
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

    # --- Ray 初始化 ---
    if ray.is_initialized():
        ray.shutdown()
    
    # 显式指定 num_cpus，彻底解决 Docker 检测警告
    ray.init(
        num_cpus=total_cores, 
        ignore_reinit_error=True, 
        log_to_driver=False, 
        include_dashboard=False
    )

    register_env("delivery_pz_env", env_creator)

    # --- 获取空间信息 (Dummy Env) ---
    # 只需要实例化一次获取 space 即可
    temp_env = DeliveryUAVEnv(max_steps=args.max_steps, seed=0)
    rider_obs_space = temp_env.observation_spaces["rider_0"]
    rider_act_space = temp_env.action_spaces["rider_0"]
    station_obs_space = temp_env.observation_spaces["station_0"]
    station_act_space = temp_env.action_spaces["station_0"]
    temp_env.close() # 记得关闭

    policies = {
        "rider_policy": (None, rider_obs_space, rider_act_space, {}),
        "station_policy": (None, station_obs_space, station_act_space, {}),
    }

    # --- PPO 参数配置 ---
    # 建议：Batch Size 设为总并发数的整数倍，或者足够大以覆盖多条轨迹
    train_batch_size = 120000 
    sgd_minibatch_size = 8192 

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="delivery_pz_env",
            env_config={"max_steps": args.max_steps, "seed": args.seed},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(
            ####=========================####           
            batch_mode="truncate_episodes",
            rollout_fragment_length='auto',
            ####=========================####
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            sample_timeout_s=600,
            num_cpus_per_env_runner=1,

        )
        .resources(
            num_gpus=num_gpus,
        )
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=train_batch_size, 
            minibatch_size=sgd_minibatch_size, 
            num_epochs=5, 
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,

            model={
                "fcnet_hiddens": [64, 64],  # 原来是 [256, 256]
                "fcnet_activation": "tanh",
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["rider_policy", "station_policy"],
        )
    )

    algo = config.build_algo()

    # --- 训练循环 ---
    log_filename = "training_log_optimized.csv"
    os.makedirs("./checkpoints", exist_ok=True)
    
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Reward_Mean", "Episode_Len_Mean", "Total_Timesteps", "Steps_Per_Sec"])
        
        print(f"--- 训练开始! 日志: {log_filename} ---")
        
        for i in range(args.iters):
            result = algo.train()
            
            # 提取指标
            metrics = result.get("env_runners", {}) or result
            mean_rew = metrics.get("episode_reward_mean", float('nan'))
            mean_len = metrics.get("episode_len_mean", 0)
            total_steps = result.get("num_env_steps_sampled", 0)
            # 这里的 throughput 包含了 采样+训练 的综合速度
            fps = result.get("num_env_steps_sampled_throughput_per_sec", 0)

            timers = result.get("timers", {})
            sample_time = timers.get("env_runner_sampling_timer", 0) # 采样耗时
            learn_time = timers.get("learner_grad_update_timer", 0)  # GPU训练耗时
            synch_time = timers.get("synch_weights_timer", 0)        # 权重同步耗时

            print(f"Iter {i+1:03d} | FPS: {fps:.0f} | Rew: {mean_rew:.2f}")
            print(f"    🔍 原始计时数据 (Debug): {timers}")

            #print(f"Iter {i+1:03d}/{args.iters} | Reward: {mean_rew:.2f} | FPS: {fps:.0f} | TotalSteps: {total_steps}")
            writer.writerow([i+1, mean_rew, mean_len, total_steps, fps])
            f.flush()

            if (i + 1) % 10 == 0: 
                save_dir = os.path.abspath(f"./checkpoints/iter_{i+1:04d}")
                algo.save(checkpoint_dir=save_dir)
                print(f"    --> 模型已保存")

    print("--- 训练结束 ---")
    ray.shutdown()

if __name__ == "__main__":
    main()