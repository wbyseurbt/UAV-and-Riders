import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import sys
import os
from ray.rllib.policy.policy import Policy

# 导入你提供的环境
try:
    from env import DeliveryUAVEnv
except ImportError:
    print("Error: 找不到 env.py，请确保文件在同一目录下。")
    exit()



class TrainedAgent:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Loading Policies from {model_path} ...")
        
        # 拼接策略路径 (假设标准 Ray 目录结构)
        rider_ckpt_path = os.path.join(model_path, "policies", "rider_policy")
        station_ckpt_path = os.path.join(model_path, "policies", "station_policy")

        try:
            # 分别加载两个角色的策略
            self.rider_policy = Policy.from_checkpoint(rider_ckpt_path)
            self.station_policy = Policy.from_checkpoint(station_ckpt_path)
            print("策略 (Policies) 加载成功！")
        except Exception as e:
            print(f"加载失败: {e}")
            print(f"请检查路径下是否有 'policies/rider_policy' 和 'policies/station_policy'")
            # 如果加载失败，这里可以抛出异常或者做 fallback，这里选择退出
            sys.exit(1)

    def compute_actions(self, env, obs):
        """
        根据当前环境的 obs 计算所有 agent 的动作
        """
        actions = {}
        
        # 遍历所有 agent
        for agent_id in env.agents:
            # 如果 agent 不在 obs 里 (可能已经 done 或者还没出现)，跳过
            if agent_id not in obs:
                continue
                
            agent_obs = obs[agent_id]
            
            if "rider" in agent_id:
                # 返回 (action, state, info)，我们只要 action (索引0)
                act = self.rider_policy.compute_single_action(agent_obs)[0]
                actions[agent_id] = act
                
            elif "station" in agent_id:
                act = self.station_policy.compute_single_action(agent_obs)[0]
                actions[agent_id] = act
                
        return actions

# ===============================================================
# 初始化绘图
# ===============================================================
fig, ax = plt.subplots(figsize=(9, 9)) # 稍微调大一点

def init_map(grid_size):
    ax.clear()
    # 绘制淡灰色网格
    for x in range(grid_size + 1):
        ax.plot([x, x], [0, grid_size], linewidth=0.5, color='#e0e0e0', zorder=0)
    for y in range(grid_size + 1):
        ax.plot([0, grid_size], [y, y], linewidth=0.5, color='#e0e0e0', zorder=0)
    
    ax.set_xlim(-1, grid_size + 1)
    ax.set_ylim(-1, grid_size + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

# ===============================================================
# 动画循环函数
# ===============================================================
def animate(frame, env, agent_model):
    init_map(env.grid_size)

    # --- 使用 RL 模型获取动作 ---
    if not hasattr(env, 'current_obs') or env.current_obs is None:
        obs, _ = env.reset()
        env.current_obs = obs

    # 计算动作
    actions = agent_model.compute_actions(env, env.current_obs)
    
    # 步进环境
    next_obs, rew, term, trunc, info = env.step(actions)
    
    # 更新 obs 给下一帧使用
    env.current_obs = next_obs

    # ===============================================================
    # 1. 绘制 站点 (Stations) - 红色三角形 & 信息标注
    # ===============================================================
    sx = [s.pos[0] for s in env.stations]
    sy = [s.pos[1] for s in env.stations]
    ax.scatter(sx, sy, c='red', marker='^', s=180, label='Station', zorder=10, edgecolors='black')
    
    for s in env.stations:
        # [修改点 1]: 显示无人机数量 (UAV) 和 等待订单数量 (Wait)
        n_uav = len(s.uav_available)
        n_wait = len(s.orders_waiting)
        
        # 格式: U:数量 / W:数量
        info_str = f"U:{n_uav}\nW:{n_wait}"
        
        # 字体颜色逻辑：如果积压订单太多(>5)，显示红色警示，否则显示深灰色
        text_color = 'red' if n_wait > 5 else '#333333'
        
        ax.text(s.pos[0], s.pos[1]+1.2, info_str, fontsize=9, ha='center', 
                color=text_color, fontweight='bold', zorder=12)

    # ===============================================================
    # 2. 绘制 餐厅/商铺 (Shops) - 橙色五角星
    # ===============================================================
    shop_x = [s[0] for s in env.shop_locs]
    shop_y = [s[1] for s in env.shop_locs]
    ax.scatter(shop_x, shop_y, c='orange', marker='*', s=120, label='Shop', alpha=0.8)

    # ===============================================================
    # 3. 绘制 骑手 (Riders)
    # ===============================================================
    r_free_x, r_free_y = [], []
    r_busy_x, r_busy_y = [], []
    
    for r in env.riders:
        if r.free:
            r_free_x.append(r.pos[0])
            r_free_y.append(r.pos[1])
        else:
            r_busy_x.append(r.pos[0])
            r_busy_y.append(r.pos[1])
            if r.target_pos is not None:
                ax.plot([r.pos[0], r.target_pos[0]], [r.pos[1], r.target_pos[1]], 
                        color='blue', linestyle=':', linewidth=0.8, alpha=0.4)
            if r.carrying_order is not None:
                # 获取订单终点
                dest = r.carrying_order.end
                # 绘制粗实线 (linewidth=2.0)
                ax.plot([r.pos[0], dest[0]], [r.pos[1], dest[1]], 
                        color='#0000ff', linestyle='-', linewidth=2.0, alpha=0.15, label='Delivery Route')
    if r_free_x: ax.scatter(r_free_x, r_free_y, c='lime', s=60, label='Rider(Free)', edgecolors='green', zorder=5)
    if r_busy_x: ax.scatter(r_busy_x, r_busy_y, c='blue', s=60, label='Rider(Busy)', edgecolors='white', zorder=5)

    # ===============================================================
    # 4. 绘制 无人机 (UAVs) - [修改点 2: 区分空载和负载]
    # ===============================================================
    u_fly_empty_x, u_fly_empty_y = [], [] # 飞行中-空载
    u_fly_load_x, u_fly_load_y = [], []   # 飞行中-负载
    u_dock_x, u_dock_y = [], []           # 停泊中
    
    for u in env.uavs:
        if u.station_id is None or u.station_id == -1: # 飞行状态
            # 判断是否携带订单
            if len(u.orders) > 0:
                u_fly_load_x.append(u.pos[0])
                u_fly_load_y.append(u.pos[1])

                for oid in u.orders:
                    # 注意：u.orders 存的是 ID，需要去 env.orders 里找对象
                    if 0 <= oid < len(env.orders):
                        target_order = env.orders[oid]
                        dest = target_order.end
                        # 绘制紫色粗线，表示这架无人机里的货是要去那里的
                        # alpha 设置低一点(0.15)，因为无人机可能带多个货，线多了会乱
                        ax.plot([u.pos[0], dest[0]], [u.pos[1], dest[1]], 
                                color='purple', linestyle='-', linewidth=1.5, alpha=0.15)
            else:
                u_fly_empty_x.append(u.pos[0])
                u_fly_empty_y.append(u.pos[1])

            # 画飞行轨迹虚线
            if u.target_station is not None and u.target_station != -1:
                target_pos = env.stations[u.target_station].pos
                ax.plot([u.pos[0], target_pos[0]], [u.pos[1], target_pos[1]], 
                        color='purple', linestyle='--', linewidth=0.8, alpha=0.3)
        else:
            u_dock_x.append(u.pos[0])
            u_dock_y.append(u.pos[1])

    # 绘制逻辑：
    # 1. 负载无人机 (Loaded): 实心深紫色菱形，看起来比较"重"
    if u_fly_load_x: 
        ax.scatter(u_fly_load_x, u_fly_load_y, c='purple', marker='D', s=80, 
                   label='UAV(Load)', zorder=8, edgecolors='black')
        
    # 2. 空载无人机 (Empty): 空心(白色填充)紫色边框菱形，看起来比较"轻"
    if u_fly_empty_x: 
        ax.scatter(u_fly_empty_x, u_fly_empty_y, c='white', marker='D', s=60, 
                   label='UAV(Empty)', zorder=8, edgecolors='purple', linewidths=1.5)
        
    # 3. 停泊无人机 (Dock): 灰色小点
    if u_dock_x: 
        ax.scatter(u_dock_x, u_dock_y, c='gray', marker='.', s=30, alpha=0.5, label='UAV(Dock)')

    # ===============================================================
    # 5. 绘制活跃订单
    # ===============================================================
    c_x = [o.end[0] for o in env.active_orders]
    c_y = [o.end[1] for o in env.active_orders]
    if c_x: ax.scatter(c_x, c_y, c='black', marker='x', s=40, alpha=0.5, label='Customer')

    # 标题与状态
    status_text = (f"Step: {env.time} | Active Orders: {len(env.active_orders)}\n"
                   f"Strategy: Trained RL Policy")
    ax.set_title(status_text)
    
    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    return []

# ===============================================================
# 主程序入口
# ===============================================================
def run():
    # -----------------------------------------------------------
    # [关键]: 请在这里填入你 checkpoints 的绝对路径
    # 必须指向包含 'policies' 文件夹的那一层，例如 'iter_0080'
    # -----------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "iter_0500") # <-- 修改这里
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误: 找不到路径 {CHECKPOINT_PATH}")
        print("请在 run.py 代码中修改 CHECKPOINT_PATH 变量。")
        return

    print("正在加载 RL 模型...")
    agent_model = TrainedAgent(CHECKPOINT_PATH)

    print("正在初始化环境...")
    env = DeliveryUAVEnv(max_steps=1000) # 可视化时可以让步数长一点
    
    # 手动 Reset 并保存初始 obs 到 env 对象上，方便 animate 函数调用
    initial_obs, _ = env.reset(seed=42)
    env.current_obs = initial_obs

    print(f"开始渲染 (地图: {env.grid_size}x{env.grid_size})...")

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(env, agent_model), # 把 model 也传进去
        frames=1000,
        interval=200,  # 刷新间隔 (毫秒)，调大一点看得更清楚
        blit=False,
        repeat=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()