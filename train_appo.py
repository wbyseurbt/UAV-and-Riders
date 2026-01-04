import argparse
import os
import csv
import logging
import warnings
import torch
import ray

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.appo import APPOConfig 
from env import DeliveryUAVEnv

# --- 环境变量 ---
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)

def env_creator(env_config):
    max_steps = env_config.get("max_steps", 200)
    seed = env_config.get("seed", None)
    env = DeliveryUAVEnv(max_steps=max_steps, seed=seed)
    return ParallelPettingZooEnv(env)

# 修改 Policy Mapping，去掉 worker 参数，改用 **kwargs
def policy_mapping_fn(agent_id, episode, **kwargs):
    if str(agent_id).startswith("rider_"):
        return "rider_policy"
    return "station_policy"



# ==========================================
# [新增] 课程学习辅助函数
# ==========================================
def get_current_prob(iteration):
    """
    计算当前强制去站点的概率 (Curriculum Schedule)
    策略:
    - 0-100 轮: 100% 强制 (让 UAV 疯狂刷数据)
    - 100-250 轮: 线性衰减 (从 1.0 降到 0.1)
    - 250+ 轮: 保持 10% (保留一点点启发式引导，或者设为0完全自主)
    """
    length_period1 = 100
    length_period2 = 250
    min_pro = 0
    if iteration < length_period1:
        return 1.0
    elif iteration < length_period2:
        # 线性插值: 随着 iter 增加，prob 减小
        return max(min_pro, 1.0 - (iteration - length_period1) / (length_period2 - length_period1) * (1-min_pro))
    else:
        return min_pro 

def update_env_prob(env, context):
    """
    这个函数会被发送到每个 Worker 里执行
    负责找到底层的 DeliveryUAVEnv 并修改概率
    """
    # 尝试穿透 ParallelPettingZooEnv 包装器找到原始环境
    base_env = getattr(env, "par_env", None) or getattr(env, "unwrapped", None) or env
    
    # 调用 env.py 中定义的接口
    if hasattr(base_env, "set_force_station_prob"):
        base_env.set_force_station_prob(context["prob"])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- 硬件配置 ---
    total_cores = 25
    num_workers = 22 
    num_envs_per_worker = 5
    
    if torch.cuda.is_available():
        print(f"--- 🚀 显卡火力全开: {torch.cuda.get_device_name(0)} ---")
        num_gpus = 1
    else:
        num_gpus = 0

    print(f"--- 🔥 APPO 极速模式 ---")
    print(f"    Workers: {num_workers} | Envs/Worker: {num_envs_per_worker}")
    
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        num_cpus=total_cores, 
        ignore_reinit_error=True, 
        log_to_driver=False, 
        include_dashboard=False
    )

    register_env("delivery_pz_env", env_creator)

    # 获取空间
    temp_env = DeliveryUAVEnv(max_steps=args.max_steps, seed=0)
    rider_obs_space = temp_env.observation_spaces["rider_0"]
    rider_act_space = temp_env.action_spaces["rider_0"]
    station_obs_space = temp_env.observation_spaces["station_0"]
    station_act_space = temp_env.action_spaces["station_0"]
    temp_env.close()

    policies = {
        "rider_policy": (None, rider_obs_space, rider_act_space, {}),
        "station_policy": (None, station_obs_space, station_act_space, {}),
    }

    # --- APPO Config ---
    config = (
        APPOConfig()
        # 禁用新 API Stack
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
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            
            # 【关键修改】APPO 不支持 'auto'，必须是整数
            rollout_fragment_length=200, 
            
            num_cpus_per_env_runner=1,
        )
        .resources(
            num_gpus=num_gpus,
        )
        .training(
            # APPO 每次更新的 Batch Size
            train_batch_size=8192, 
            entropy_coeff=0.001,
            lr=1e-4, 
            grad_clip=40.0,
            learner_queue_size=16,
            
            model={
                "fcnet_hiddens": [256, 256], 
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

    log_filename = "training_appo.csv"
    os.makedirs("./checkpoints", exist_ok=True)
    
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Reward_Mean", "Episode_Len_Mean", "Total_Timesteps", "Steps_Per_Sec"])
        
        print(f"--- 训练开始! 日志: {log_filename} ---")
        
        for i in range(args.iters):
            # ================= [新增] 动态调整概率 =================
            # 1. 计算当前轮次的概率
            current_prob = get_current_prob(i)
            
            # 2. 广播给所有 Worker (并行环境)
            # 新版 Ray 使用 env_runner_group 替代 workers
            algo.env_runner_group.foreach_env(
                lambda env: update_env_prob(env, {"prob": current_prob})
            )
            # =======================================================



            result = algo.train()
            
            metrics = result.get("env_runners", {}) or result
            mean_rew = metrics.get("episode_reward_mean", float('nan'))
            mean_len = metrics.get("episode_len_mean", 0)
            total_steps = result.get("num_env_steps_sampled", 0)
            fps = result.get("num_env_steps_sampled_throughput_per_sec", 0)
            timers = result.get("timers", {})


            print(f"Iter {i+1:03d} | FPS: {fps:.0f} | Rew: {mean_rew:.2f}")
            print(f"    🔍 Debug: {timers}")

            writer.writerow([i+1, mean_rew, mean_len, total_steps, fps])
            f.flush()

            if (i + 1) % 20 == 0: 
                save_dir = os.path.abspath(f"./checkpoints/iter_{i+1:04d}")
                algo.save(checkpoint_dir=save_dir)
                print(f"    --> 模型已保存")

    print("--- 训练结束 ---")
    ray.shutdown()

if __name__ == "__main__":
    main()