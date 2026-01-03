import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import sys

# 仅导入环境配置
try:
    from env import DeliveryUAVEnv
    import env_config as cfg
except ImportError:
    print("Error: 找不到 env.py 或 env_config.py")
    sys.exit()

# ==========================================
# 0. 状态码翻译配置 (改为纯英文，防止方框)
# ==========================================
STATUS_MAP = {
    0: "Wait",
    1: "Picking",
    2: "Station",
    3: "Flying",
    4: "Landed",
    5: "Delivering",
    6: "Done"
}
# 对应状态的颜色，与地图上的实体颜色呼应
COLORS_MAP = {
    0: "#666666", # Gray
    1: "#00cc00", # Green (Rider)
    2: "#cc0000", # Red (Station)
    3: "#800080", # Purple (UAV)
    4: "#cc0000", # Red
    5: "#0000ff", # Blue (Rider Busy)
    6: "#000000"  # Black
}

# ==========================================
# 1. 交互控制器 (保持不变)
# ==========================================
class InteractiveController:
    def __init__(self, env, ax_map):
        self.env = env
        self.ax_map = ax_map # 绑定地图坐标轴
        self.mode = "VIEW"
        self.step_click = 0 
        self.temp_data = {} 
        self.persistent_actions = {} 

        print("\n=== Env Test (UI Enhanced) ===")
        print(" [O] Order | [R] Rider | [U] UAV | [ESC] Cancel")
        print("===================================\n")

    def on_key(self, event):
        if event.key == 'o':
            self.mode = "ORDER"; self.step_click = 0; self.temp_data = {}
        elif event.key == 'r':
            self.mode = "RIDER"; self.step_click = 0; self.temp_data = {}
        elif event.key == 'u':
            self.mode = "UAV"; self.step_click = 0; self.temp_data = {}
        elif event.key == 'escape':
            self.mode = "VIEW"; self.step_click = 0; self.temp_data = {}

    def on_click(self, event):
        # [关键保护] 只有点击在左侧地图里才生效，点右侧侧边栏不触发
        if event.inaxes != self.ax_map:
            return

        if event.xdata is None or event.ydata is None: return
        gx, gy = int(round(event.xdata)), int(round(event.ydata))
        gx = max(0, min(self.env.grid_size, gx))
        gy = max(0, min(self.env.grid_size, gy))
        click_pos = np.array([gx, gy])

        if self.mode == "ORDER": self._handle_order_click(click_pos)
        elif self.mode == "RIDER": self._handle_rider_click(click_pos)
        elif self.mode == "UAV": self._handle_uav_click(click_pos)

    # --- 处理逻辑 (保持原样) ---
    def _handle_order_click(self, pos):
        if self.step_click == 0:
            self.temp_data['start'] = pos; self.step_click = 1
            print(f"  Start: {pos}")
        elif self.step_click == 1:
            self.env.add_manual_order(self.temp_data['start'], pos)
            self.mode = "VIEW"

    def _handle_rider_click(self, pos):
        if self.step_click == 0:
            nearest_rider = min(self.env.riders, key=lambda r: np.linalg.norm(r.pos - pos))
            if np.linalg.norm(nearest_rider.pos - pos) > 3: return
            self.temp_data['rider_idx'] = nearest_rider.rid; self.step_click = 1
            print(f"  Rider Selected: {nearest_rider.rid}")
        elif self.step_click == 1:
            rid = self.temp_data['rider_idx']
            target_sid = -1
            for s in self.env.stations:
                if np.array_equal(s.pos, pos): target_sid = s.sid; break
            action = target_sid + 1 if target_sid != -1 else 0
            self.persistent_actions[f"rider_{rid}"] = action
            self.mode = "VIEW"

    def _handle_uav_click(self, pos):
        if self.step_click == 0:
            for s in self.env.stations:
                if np.linalg.norm(s.pos - pos) < 1.5:
                    self.temp_data['src_sid'] = s.sid; self.step_click = 1
                    print(f"  Station Selected: {s.sid}")
                    return
        elif self.step_click == 1:
            target_sid = -1
            for s in self.env.stations:
                if np.linalg.norm(s.pos - pos) < 1.5: target_sid = s.sid; break
            if target_sid == -1 or target_sid == self.temp_data['src_sid']: return
            src_sid = self.temp_data['src_sid']
            src_st = self.env.stations[src_sid]
            uav_pick = 1 if src_st.uav_available else 0
            order_pick = 1 if src_st.orders_waiting else 0
            if uav_pick == 0: self.mode = "VIEW"; return

            single_act = [uav_pick, target_sid, order_pick] + [0]*(self.env.uav_cap - 1)
            full_act = np.array(single_act + [0]*len(single_act)*(self.env.concurrent_launches-1))
            self.persistent_actions[f"station_{src_sid}"] = full_act
            self.mode = "VIEW"

    def get_actions(self):
        actions = self.persistent_actions.copy()
        keys_to_remove = [k for k in actions.keys() if k.startswith("station_")]
        for k in keys_to_remove: del self.persistent_actions[k]
        return actions
    
    def clear_rider_action(self, agent_id):
        if agent_id in self.persistent_actions: del self.persistent_actions[agent_id]

# ==========================================
# 2. 布局初始化 (保持左侧不变)
# ==========================================
fig = plt.figure(figsize=(14, 9)) 

gs = GridSpec(1, 2, width_ratios=[1, 0.4], wspace=0.1)

# 左边是地图
ax_map = fig.add_subplot(gs[0])
# 右边是信息栏
ax_info = fig.add_subplot(gs[1])

def init_display(grid_size):
    # --- 初始化左侧地图 ---
    ax_map.clear()
    for x in range(grid_size + 1):
        ax_map.plot([x, x], [0, grid_size], lw=0.5, color='#e0e0e0', zorder=0)
    for y in range(grid_size + 1):
        ax_map.plot([0, grid_size], [y, y], lw=0.5, color='#e0e0e0', zorder=0)
    
    ax_map.set_xlim(-1, grid_size + 1)
    ax_map.set_ylim(-1, grid_size + 1)
    ax_map.set_aspect("equal") 
    
    # --- 初始化右侧信息栏 (移除 Emoji) ---
    ax_info.clear()
    ax_info.set_axis_off() 
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    
    # 纯英文标题
    ax_info.text(0, 0.96, "ORDER DASHBOARD", fontsize=14, fontweight='bold', color='#333333')
    ax_info.plot([0, 1], [0.95, 0.95], color='#aaaaaa', lw=1.5)

# ==========================================
# 3. 动画循环
# ==========================================
def animate(frame, env, controller):
    init_display(env.grid_size)

    # --- 逻辑更新 ---
    is_paused = (controller.mode != "VIEW")
    if not is_paused:
        actions = {}
        for agent_id in env.rider_agents: actions[agent_id] = 0
        for agent_id in env.station_agents:
            dims = env.action_space(agent_id).nvec
            actions[agent_id] = np.zeros(dims.shape, dtype=int)
        
        user_cmds = controller.get_actions()
        if user_cmds: actions.update(user_cmds)
        
        obs, rew, term, trunc, info = env.step(actions)
        for r in env.riders:
            if r.free: controller.clear_rider_action(f"rider_{r.rid}")

    # ==========================================
    # A. 绘制左侧地图 (完全保留上一版样式)
    # ==========================================
    # 站点
    sx = [s.pos[0] for s in env.stations]; sy = [s.pos[1] for s in env.stations]
    ax_map.scatter(sx, sy, c='red', marker='^', s=180, label='Station', zorder=10, ec='black')
    for s in env.stations:
        n_wait = len(s.orders_waiting)
        col = 'red' if n_wait > 0 else '#333333'
        ax_map.text(s.pos[0], s.pos[1]+1.2, f"U:{len(s.uav_available)}\nW:{n_wait}", 
                    fontsize=9, ha='center', color=col, fontweight='bold', zorder=12)

    # 商店
    shop_x = [s[0] for s in env.shop_locs]; shop_y = [s[1] for s in env.shop_locs]
    ax_map.scatter(shop_x, shop_y, c='orange', marker='*', s=120, label='Shop', alpha=0.8)

    # 骑手
    r_free_x, r_free_y = [], []; r_busy_x, r_busy_y = [], []
    for r in env.riders:
        if r.free: r_free_x.append(r.pos[0]); r_free_y.append(r.pos[1])
        else:
            r_busy_x.append(r.pos[0]); r_busy_y.append(r.pos[1])
            if r.target_pos is not None:
                ax_map.plot([r.pos[0], r.target_pos[0]], [r.pos[1], r.target_pos[1]], color='blue', ls=':', lw=0.8, alpha=0.4)
    if r_free_x: ax_map.scatter(r_free_x, r_free_y, c='lime', s=60, label='Free', ec='green', zorder=5)
    if r_busy_x: ax_map.scatter(r_busy_x, r_busy_y, c='blue', s=60, label='Busy', ec='white', zorder=5)

    # 无人机
    u_load_x, u_load_y = [], []
    u_empty_x, u_empty_y = [], []
    u_dock_x, u_dock_y = [], []
    for u in env.uavs:
        if u.station_id is None:
            if u.orders: u_load_x.append(u.pos[0]); u_load_y.append(u.pos[1])
            else: u_empty_x.append(u.pos[0]); u_empty_y.append(u.pos[1])
            if u.target_station is not None:
                tp = env.stations[u.target_station].pos
                ax_map.plot([u.pos[0], tp[0]], [u.pos[1], tp[1]], color='purple', ls='--', lw=0.8, alpha=0.3)
        else: u_dock_x.append(u.pos[0]); u_dock_y.append(u.pos[1])

    if u_load_x: ax_map.scatter(u_load_x, u_load_y, c='purple', marker='D', s=80, label='UAV(Load)', zorder=8, ec='black')
    if u_empty_x: ax_map.scatter(u_empty_x, u_empty_y, c='white', marker='D', s=60, label='UAV(Empty)', zorder=8, ec='purple', lw=1.5)
    if u_dock_x: ax_map.scatter(u_dock_x, u_dock_y, c='gray', marker='.', s=30, alpha=0.5, label='UAV(Dock)')

    # 订单位置
    c_x = [o.end[0] for o in env.active_orders]; c_y = [o.end[1] for o in env.active_orders]
    if c_x: ax_map.scatter(c_x, c_y, c='black', marker='x', s=40, alpha=0.5)

    # 地图标题
    title_text = f"[{'PAUSED' if is_paused else 'RUNNING'}] Mode: {controller.mode} | Time: {env.time}"
    ax_map.set_title(title_text, color="red" if is_paused else "green", fontweight='bold')

    # ==========================================
    # B. 绘制右侧列表 (改为纯英文)
    # ==========================================
    y_pos = 0.90
    line_spacing = 0.08
    
    if not env.active_orders:
        ax_info.text(0.5, 0.5, "No Active Orders\n(Press 'O' to add)", ha='center', va='center', color='gray', fontsize=12)
    else:
        count = 0
        for o in reversed(env.active_orders):
            if count >= 10:
                ax_info.text(0.5, y_pos, "... More ...", ha='center', color='gray')
                break
            
            # 状态纯英文
            status_txt = STATUS_MAP.get(o.status, "Unknown")
            color_code = COLORS_MAP.get(o.status, "black")
            
            # 第一行：ID 和 状态
            ax_info.text(0, y_pos, f"Order #{o.oid}", fontsize=11, fontweight='bold', color='#333333')
            ax_info.text(0.35, y_pos, f"[{status_txt}]", fontsize=10, fontweight='bold', color=color_code)
            
            # 第二行：路径
            route_str = f"Route: {o.start} -> {o.end}"
            ax_info.text(0, y_pos - 0.03, route_str, fontsize=9, color='#555555')
            
            # 第三行：负责人 (纯英文，无Emoji)
            wait_str = f"Wait: {o.time_wait} min"
            handler_str = ""
            if o.status == 1: handler_str = f"Rider {o.rider1_id}"
            elif o.status == 3: handler_str = f"UAV {o.uav_id}"
            elif o.status == 5: handler_str = f"Rider {o.rider2_id}"
            
            info_line = f"{wait_str}   {handler_str}"
            ax_info.text(0, y_pos - 0.055, info_line, fontsize=9, color='blue' if handler_str else '#777777')
            
            # 分割线
            ax_info.plot([0, 1], [y_pos - 0.07, y_pos - 0.07], color='#eeeeee', lw=1)
            
            y_pos -= line_spacing
            count += 1

    return []

def run():
    print("Initializing UI Enhanced Environment...")
    env = DeliveryUAVEnv(max_steps=10000) 
    env._generate_orders = lambda: None 
    env.reset()
    
    controller = InteractiveController(env, ax_map)
    fig.canvas.mpl_connect('key_press_event', controller.on_key)
    fig.canvas.mpl_connect('button_press_event', controller.on_click)

    ani = animation.FuncAnimation(fig, animate, fargs=(env, controller),
                                  frames=10000, interval=100, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()