import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# 导入你提供的环境
try:
    from env import DeliveryUAVEnv
except ImportError:
    print("Error: 找不到 env.py，请确保文件在同一目录下。")
    exit()

# ===============================================================
# 1. 初始化绘图
# ===============================================================
# 设置较大的画布以便观察
fig, ax = plt.subplots(figsize=(8, 8))

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
# 2. 简单的规则策略 (Heuristic Policy)
# ===============================================================
def heuristic_policy(env):
    """
    一个简单的规则策略，用于演示环境运行：
    1. Station: 如果有空闲UAV和积压订单，就发射去最近的邻居站点。
    2. Rider: 如果空闲，就去最近的站点待命（等待被自动分配最后一公里任务）。
    """
    actions = {}

    # --- A. 骑手策略 ---
    # 你的 env.py 中 _handle_rider_last_mile 会自动把订单塞给空闲骑手
    # 所以骑手只需要保持空闲并靠近站点即可
    for i, agent in enumerate(env.rider_agents):
        rider = env.riders[i]
        
        if rider.carrying_order:
            # 如果手上有单，动作0 = 也就是走向终点（env内部处理）
            actions[agent] = 0
        else:
            # 如果空闲，随机去一个站点待命，增加被分配订单的概率
            # 动作定义: 0=直送/待命, 1~N=去站点
            target_sid = (rider.rid % env.n_stations) # 简单的分配逻辑
            actions[agent] = target_sid + 1 

    # --- B. 站点策略 (调度 UAV) ---
    for k, agent in enumerate(env.station_agents):
        station = env.stations[k]
        
        # 获取动作空间的维度
        dims = env.action_spaces[agent].nvec
        # 初始化全0向量 (NoOp)
        action_vec = np.zeros(dims.shape, dtype=int)
        
        # 简单的贪心逻辑：尝试填满并发发射槽位
        # 复制列表以免影响循环内的逻辑判断
        available_uavs = station.uav_available[:] 
        waiting_orders = station.orders_waiting[:]
        
        for slot in range(env.concurrent_launches):
            base_idx = slot * env.single_action_len
            
            # 只有当 既有飞机 又有订单 时才发射
            if available_uavs and waiting_orders:
                # 1. 选飞机 (Global ID)
                uav_id = available_uavs.pop(0)
                
                # 2. 选订单 (Global ID)
                order_id = waiting_orders.pop(0)
                order_obj = env.orders[order_id]
                
                # 3. 选目标: 找离订单终点最近的另一个站点
                best_target = -1
                min_dist = float('inf')
                for other_s in env.stations:
                    if other_s.sid == station.sid: continue
                    dist = np.linalg.norm(other_s.pos - order_obj.end)
                    if dist < min_dist:
                        min_dist = dist
                        best_target = other_s.sid
                
                if best_target == -1: 
                    best_target = (station.sid + 1) % env.n_stations
                
                # 填入动作 (注意 env.py 要求的是 Global ID)
                # [0] UAV ID
                action_vec[base_idx] = uav_id
                # [1] Target Station ID
                action_vec[base_idx + 1] = best_target
                # [2] Order ID
                action_vec[base_idx + 2] = order_id
                
                # 后面的订单槽位留空 (假设一次运一单)
            
        actions[agent] = action_vec

    return actions

# ===============================================================
# 3. 动画循环
# ===============================================================
def animate(frame, env):
    init_map(env.grid_size)

    # --- 获取动作并步进环境 ---
    actions = heuristic_policy(env)
    obs, rew, term, trunc, info = env.step(actions)

    # ================= 绘制实体 =================

    # 1. 绘制 站点 (Stations) - 红色三角形
    sx = [s.pos[0] for s in env.stations]
    sy = [s.pos[1] for s in env.stations]
    ax.scatter(sx, sy, c='red', marker='^', s=150, label='Station', zorder=10, edgecolors='black')
    
    # 显示站点信息 (等待UAV取货数量 / 等待骑手送货数量)
    for s in env.stations:
        info_str = f"Wait:{len(s.orders_waiting)}\nDeliv:{len(s.orders_to_deliver)}"
        ax.text(s.pos[0], s.pos[1]+1.5, info_str, fontsize=8, ha='center', color='darkred', fontweight='bold')

    # 2. 绘制 餐厅/商铺 (Shops) - 橙色五角星
    shop_x = [s[0] for s in env.shop_locs]
    shop_y = [s[1] for s in env.shop_locs]
    ax.scatter(shop_x, shop_y, c='orange', marker='*', s=120, label='Shop', alpha=0.8)

    # 3. 绘制 骑手 (Riders) - 圆形
    # 空闲骑手(绿色)，忙碌骑手(蓝色)
    r_free_x, r_free_y = [], []
    r_busy_x, r_busy_y = [], []
    
    for r in env.riders:
        if r.free:
            r_free_x.append(r.pos[0])
            r_free_y.append(r.pos[1])
        else:
            r_busy_x.append(r.pos[0])
            r_busy_y.append(r.pos[1])
            # 画出忙碌骑手到目标的连线
            if r.target_pos is not None:
                ax.plot([r.pos[0], r.target_pos[0]], [r.pos[1], r.target_pos[1]], 
                        color='blue', linestyle=':', linewidth=0.8, alpha=0.4)

    if r_free_x: ax.scatter(r_free_x, r_free_y, c='lime', s=50, label='Rider(Free)', edgecolors='green', zorder=5)
    if r_busy_x: ax.scatter(r_busy_x, r_busy_y, c='blue', s=50, label='Rider(Busy)', edgecolors='white', zorder=5)

    # 4. 绘制 无人机 (UAVs) - 菱形
    # 飞行中(紫色)，停泊中(灰色)
    u_fly_x, u_fly_y = [], []
    u_dock_x, u_dock_y = [], []
    
    for u in env.uavs:
        if u.station_id is None or u.station_id == -1: # 飞行中
            u_fly_x.append(u.pos[0])
            u_fly_y.append(u.pos[1])
            # 画飞行轨迹线（指向目标站点）
            if u.target_station is not None and u.target_station != -1:
                target_pos = env.stations[u.target_station].pos
                ax.plot([u.pos[0], target_pos[0]], [u.pos[1], target_pos[1]], 
                        color='purple', linestyle='--', linewidth=0.8, alpha=0.3)
        else:
            u_dock_x.append(u.pos[0])
            u_dock_y.append(u.pos[1])

    if u_fly_x: ax.scatter(u_fly_x, u_fly_y, c='purple', marker='D', s=60, label='UAV(Fly)', zorder=8)
    if u_dock_x: ax.scatter(u_dock_x, u_dock_y, c='gray', marker='.', s=30, alpha=0.5, label='UAV(Dock)')

    # 5. 绘制活跃订单终点 (Customers) - 黑色叉号
    # 避免绘制太多，只绘制 active 的
    c_x = [o.end[0] for o in env.active_orders]
    c_y = [o.end[1] for o in env.active_orders]
    if c_x: ax.scatter(c_x, c_y, c='black', marker='x', s=30, alpha=0.3, label='Customer')

    # 标题与图例
    ax.set_title(f"Step: {env.time} | Active Orders: {len(env.active_orders)}")
    
    # 简单的图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    return []

# ===============================================================
# 4. 主程序入口
# ===============================================================
def run():
    print("正在初始化环境...")
    env = DeliveryUAVEnv()
    env.reset()
    print("环境初始化完成，开始渲染...")
    print(f"地图大小: {env.grid_size}x{env.grid_size}, 骑手: {env.n_riders}, 无人机: {env.n_uavs}")

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(env,),
        frames=1000,
        interval=100,  # 刷新间隔 (毫秒)
        blit=False,
        repeat=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()