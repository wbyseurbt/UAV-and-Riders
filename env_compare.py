import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ray
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy

import env_config as cfg 
# 你的原始强化学习环境
from env import DeliveryUAVEnv 
# 纯骑手对照环境
from env_pure_rider import PureRiderEnv 


# ==========================================
# 0. 注册环境函数
# ==========================================
def env_creator(env_config):
    # 保持和训练时一致的参数
    return DeliveryUAVEnv(env_config)

ENV_NAME = "delivery_pz_env" 
register_env(ENV_NAME, env_creator)

# ==========================================
# 1. 模拟你的模型 (Policy 模式)
# ==========================================
class TrainedAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        print(f"Loading Policies from {model_path} ...")
        
        # 拼接策略路径
        rider_ckpt_path = os.path.join(model_path, "policies", "rider_policy")
        station_ckpt_path = os.path.join(model_path, "policies", "station_policy")

        try:
            # [关键修复 2]: 分别加载两个角色的策略权重
            self.rider_policy = Policy.from_checkpoint(rider_ckpt_path)
            self.station_policy = Policy.from_checkpoint(station_ckpt_path)
            print("✅ 策略 (Policies) 加载成功！")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print(f"请检查路径下是否有 'policies/rider_policy' 和 'policies/station_policy'")
            raise e

    def compute_action(self, agent_id, obs):
        """
        输入: agent_id (str), obs (np.array)
        输出: action (int or list)
        """
        # [关键修复 3]: 根据名字调用对应的策略对象
        if "rider" in agent_id:
            # result = (action, state_out, info)
            result = self.rider_policy.compute_single_action(obs)
            return result[0] # 只返回动作
            
        elif "station" in agent_id:
            result = self.station_policy.compute_single_action(obs)
            return result[0]
            
        else:
            return 0 # Fallback

# ==========================================
# 2. 评估核心逻辑
# ==========================================
def run_evaluation(env_class, model=None, episodes=5, label="Default"):
    print(f"--- 正在评估: {label} ---")
    
    metrics = {
        "total_reward": [],
        "orders_delivered": [],
        "avg_wait_time": []
    }

    for ep in tqdm(range(episodes)):
        # 建议把 max_steps 改回 200 以匹配训练设置，当然 300 也可以
        env = env_class(max_steps=200) 
        obs, _ = env.reset(seed=42 + ep) 
        
        episode_reward = 0
        done = False
        
        while not done:
            actions = {}
            
            # --- A. 纯骑手模式 (Baseline) ---
            if model is None:
                for agent in env.agents:
                    actions[agent] = 0
            
            # --- B. 强化学习模式 (AI) ---
            else:
                for agent in env.agents:
                    if agent in obs: 
                        act = model.compute_action(agent, obs[agent])
                        actions[agent] = act
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            if len(env.agents) > 0:
                first_agent = list(rewards.keys())[0]
                episode_reward += rewards[first_agent]
            
            done = all(terms.values()) or all(truncs.values())

        # 统计本局指标
        success_code = 6 if label == "RL + UAV" else 2
        delivered_count = sum(1 for o in env.orders if o.status == success_code)
        all_waits = [o.time_wait for o in env.orders]
        avg_wait = np.mean(all_waits) if all_waits else 0

        metrics["total_reward"].append(episode_reward)
        metrics["orders_delivered"].append(delivered_count)
        metrics["avg_wait_time"].append(avg_wait)

    summary = {k: np.mean(v) for k, v in metrics.items()}
    return summary

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    
    # 初始化 Ray (去除 local_mode，正常初始化)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # [关键修复 4]: 使用刚才 debug 确认过的真实路径
    # 使用原始字符串 r"" 防止反斜杠转义
    CHECKPOINT_PATH = r"D:\Study\Professional_Knowledge\RL\UAV-and-Riders\checkpoints\iter_0280"

    print("准备开始对比测试...")

    # 1. 初始化智能体
    try:
        my_ai_agent = TrainedAgent(model_path=CHECKPOINT_PATH)
    except Exception as e:
        print(f"错误: {e}")
        exit()

    # 2. 运行对比
    # 跑基准
    baseline_stats = run_evaluation(PureRiderEnv, model=None, episodes=10, label="Pure Rider")
    
    # 跑模型
    rl_stats = run_evaluation(DeliveryUAVEnv, model=my_ai_agent, episodes=10, label="RL + UAV")

    # 3. 打印表格
    print("\n" + "="*60)
    print(f"{'指标 (Metric)':<25} | {'纯骑手 (Baseline)':<15} | {'RL + 无人机 (Ours)':<15}")
    print("-" * 65)
    
    for key in baseline_stats.keys():
        val1 = baseline_stats[key]
        val2 = rl_stats[key]
        # 避免除以0
        diff = ((val2 - val1) / (abs(val1) + 1e-6)) * 100
        print(f"{key:<27} | {val1:15.2f} | {val2:15.2f} ({diff:+.1f}%)")
    print("="*60)

    # 4. 绘图
    labels = ["Total Reward", "Orders Delivered", "Wait Time (min)"]
    base_vals = [baseline_stats["total_reward"], baseline_stats["orders_delivered"], baseline_stats["avg_wait_time"]]
    rl_vals = [rl_stats["total_reward"], rl_stats["orders_delivered"], rl_stats["avg_wait_time"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Pure Rider', color='gray', alpha=0.6)
    rects2 = ax.bar(x + width/2, rl_vals, width, label='RL + UAV', color='#1f77b4')

    ax.set_ylabel('Value')
    ax.set_title('Comparison: Pure Rider vs RL-UAV')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('eval_result.png')
    print("\n结果图表已保存为 eval_result.png")
    plt.show()

    ray.shutdown()