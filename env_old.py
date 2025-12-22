import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import random
import env_config as cfg


# ================================================================
# 订单状态定义
# ================================================================
ORDER_STATUS = {
    "UNASSIGNED":       0,
    "PICKED_BY_R1":     1, # 骑手1已取餐，正在前往无人机站点
    "AT_STATION":       2, # 到达无人机站点，等待无人机
    "IN_UAV":           3, # 无人机运输中
    "AT_DROP_POINT":    4, # 无人机到达降落点
    "PICKED_BY_R2":     5, # 骑手2取货，前往终点
    "DELIVERED":        6, # 完成
}

# ================================================================
# 订单类
# ================================================================
class Order:
    def __init__(self, oid, start, end, time):
        self.oid = oid
        self.start = start  # 取货点（商店位置）
        self.end = end      # 送货点（客户位置）
        self.time_created = time
        self.time_wait = 0

        self.status = ORDER_STATUS["UNASSIGNED"]

        self.rider1_id = None
        self.station1_id = None
        self.uav_id = None
        self.station2_id = None
        self.rider2_id = None

# ================================================================
# 骑手类
# ================================================================
class Rider:
    def __init__(self, rid, pos):
        self.rid = rid
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None

        # 速度控制：0.5格/min
        # 实现方式：累积器。每step增加0.5，>=1时才允许发生实际位移
        self.move_buffer = 0.0 
        self.speed = cfg.Rider_SPEED
    def reset(self):
        self.free = True
        self.carrying_order = None
        self.target_pos = None
        self.move_buffer = 0.0

# ================================================================
# 无人机类
# ================================================================
class UAV:
    def __init__(self, uid, pos, station_id):
        self.speed = cfg.UAV_SPEED
        self.capacity_limit = cfg.UAV_capacity_limit

        self.uid = uid
        self.pos = np.array(pos, dtype=float)

        self.battery = 1.0 

        self.state = "STOP" # 状态：STOP, CHARGING, FLYING
        self.orders = [] # 当前搭载的订单列表,注意，是ID列表
        self.station_id = station_id  # 当前所在站点ID
        self.target_station = None    # 目标站点ID，-1表示未飞行

    @property
    def is_full(self):
        return len(self.orders) >= self.capacity_limit
    

# ================================================================
# 站点类
# ================================================================
class Station:
    def __init__(self, sid, pos):
        self.sid = sid
        self.max_uavs = cfg.Station_MAX_UAVS
        self.max_order_buffer = cfg.Station_MAX_ORDER_BUFFER
        self.pos = np.array(pos, dtype=float)
        
        self.orders_to_deliver = []  # 待骑手取货订单列表，注意是ID列表
        self.uav_available = []     # 可用无人机列表，注意是ID列表
        self.orders_waiting = []    # 等待无人机取货的订单列表，注意是ID列表
# ================================================================
# ====================== ENVIRONMENT =============================
# ================================================================
class DeliveryUAVEnv(ParallelEnv):
    metadata = {"name": "delivery_uav_v1"}

    # ------------------------------------------------------------
    def __init__(self):
        super().__init__()
        #四大重要参数self.uavs self.riders self.stations self.orders

        # ===== 环境参数 =====
        self.grid_size = cfg.World_grid_size
        self.n_riders = cfg.World_n_riders
        self.n_uavs = cfg.World_n_uavs
        self.n_stations = cfg.World_n_stations
        self.n_shops = cfg.World_n_shops
        self.station_locs = [np.array(pos) for pos in cfg.World_locs_stations]
        self.shop_locs = [np.array(pos) for pos in cfg.World_locs_shops]

        # 限制参数（用于定义固定大小的 Action Space）
        self.concurrent_launches = cfg.Station_MAX_CONCURRENT_LAUNCH # 例如 2


        
        # ===== Agent Names =====
        self.rider_agents = [f"rider_{i}" for i in range(self.n_riders)]
        self.station_agents = [f"station_{k}" for k in range(self.n_stations)]
        self.agents = self.rider_agents + self.station_agents

        # ============================================================
        # ===== Action Space Definition (关键修改) =====
        # ============================================================
        self.action_spaces = {}
        # --- 1. Rider Action Space ---
        # 动作: [0: 直接送, 1: 去站点0, 2: 去站点1, ..., N: 去站点N-1]
        # 含义: 决定当前订单的处理策略
        for agent in self.rider_agents:
            self.action_spaces[agent] = spaces.Discrete(1 + self.n_stations)

        # --- 2. Station Action Space ---
        # 这是一个复杂的组合动作，使用 MultiDiscrete
        # 维度含义:
        # [0]: 选哪个 UAV (None 或者<0不发射, 0~N=self.uavs中的第i个UAV)
        # [1]: 去哪个站点 (0~M-1)
        # [2]~[2+Cap]: 订单槽位1~5，选择self.orders的全局索引 ((None 或者<0不发射, 0~K=self.orders中的第i个订单)
        # 定义 "单个发射指令" 的维度结构
        # [UAV选择, 目标站点, 订单1, ..., 订单Capacity]
        uav_cap = cfg.UAV_capacity_limit

        single_launch_dims = (
            [1] +
            [1] +
            [1] * uav_cap  
        )
        # 语法解释[self.max_uavs_per_station + 1]: 产生一个包含一个元素的列表
        
        # 将 "单个指令" 重复 N 次 (concurrent_launches)
        # 例如: Launch_1 + Launch_2
        full_station_dims = single_launch_dims * self.concurrent_launches
        
        # 记录切片长度以便 step 解析
        self.single_action_len = len(single_launch_dims)

        for agent in self.station_agents:
            self.action_spaces[agent] = spaces.MultiDiscrete(full_station_dims)

        # ===== Observation Space Definition =====      

        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1000, shape=(200,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.time = 0
        self.orders = []
        self.active_orders = []
        
        self.stations = [Station(i, pos) for i, pos in enumerate(self.station_locs)]
        self.uavs = []
        
        # 初始分配 UAV
        for i in range(self.n_uavs):
            sid = i % self.n_stations
            uav = UAV(i, self.stations[sid].pos, sid)
            self.uavs.append(uav)
            self.stations[sid].uav_available.append(i)

        self.riders = [
            Rider(i, [random.randint(0, self.grid_size), random.randint(0, self.grid_size)]) 
            for i in range(self.n_riders)
        ]

        return {agent: self._get_obs(agent) for agent in self.agents}, {}

    def step(self, actions):
        self.time += 1
        for order in self.active_orders:
            order.time_wait += 1
        # 生成订单
        self._generate_orders()

        # 处理 Station 动作 (重点修改)
        for agent_name, action in actions.items():
            if "station" in agent_name:
                sid = int(agent_name.split("_")[1])
                self._process_station_action(sid, action)

        # 处理 Rider 动作
        for agent_name, action in actions.items():
            if "rider" in agent_name:
                rid = int(agent_name.split("_")[1])
                self._process_rider_action(rid, action)

        for order in self.active_orders:
            if order.status in [ORDER_STATUS["AT_DROP_POINT"]]:
                station = self.stations[order.station2_id]
                self._handle_rider_last_mile(station)
                
        # 物理移动
        self._move_riders()
        self._move_uavs()

        # 返回
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents} 
        terminated = {a: self.time >= 2000 for a in self.agents}
        truncated = {a: False for a in self.agents}
        return obs, rewards, terminated, truncated, {}

    # ================================================================
    # 核心逻辑: Station 批量处理
    # ================================================================
    def _process_station_action(self, sid, action_vec):
        station = self.stations[sid]
        
        # 临时记录本回合已占用的资源，防止并发冲突
        used_uav_indices = set()
        used_order_indices = set()

        # 循环处理每一条发射指令 (slot 0 ~ K-1)
        for k in range(self.concurrent_launches):
            # 1. 切片提取当前 slot 的动作向量
            start_idx = k * self.single_action_len
            end_idx = (k + 1) * self.single_action_len
            sub_action = action_vec[start_idx : end_idx]
            
            # 2. 解析动作
            uav_choice_idx = sub_action[0]  # -1 表示不发射
            target_sid = sub_action[1]
            order_choices = sub_action[2:]
            
            ### --- 校验逻辑 --- ####
            
            # A. 检查 UAV 选择是否有效
            if uav_choice_idx < 0 or uav_choice_idx is None:
                continue # 这个 slot 选择了不发射
            
            # 检查重复使用：如果这个 UAV 在上一个 slot 被选了
            if uav_choice_idx in used_uav_indices:
                continue # 无效动作：UAV已被占用
            
            ##########################！！！！！！！！！！！！！！！！！！
            #注意uav_choice_idx是全局uavs索引，而station.uav_available是局部索引
            #注意buf_idx是全局orders索引，而station.orders_waiting是局部索引
            ##########################！！！！！！！！！！！！！！！！！！
            if uav_choice_idx not in station.uav_available:
                continue # 索引越界
            
            uav_id = uav_choice_idx
            uav = self.uavs[uav_id]
            
            if uav.battery < 0.2:
                continue # 电量不足
            
            # B. 检查订单选择是否有效
            orders_to_load = []
            
            # 收集要装载的订单 buffer index
            valid_order_buffer_indices = []
            for buf_idx in order_choices:
                if buf_idx < 0 or buf_idx is None: continue # 选择了不装载

                if buf_idx not in station.orders_waiting:
                    continue
                # 检查重复使用 (同一架飞机选两遍，或被上架飞机选走了)
                if buf_idx in used_order_indices:
                    continue
                
                valid_order_buffer_indices.append(buf_idx)
            
            if not valid_order_buffer_indices:
                continue # 没有选到任何有效订单，取消本次发射

            
            # 标记 UAV 已使用
            used_uav_indices.add(uav_choice_idx)
            
            # 修改订单状态
            for buf_idx in valid_order_buffer_indices:
                order = self.orders[buf_idx]
                order.status = ORDER_STATUS["IN_UAV"]
                order.uav_id = uav.uid
                orders_to_load.append(buf_idx)
                used_order_indices.add(buf_idx)
                ########self.orders[buf_idx] = order  # 覆盖
            # 配置 UAV
            uav.orders = orders_to_load
            uav.station_id = None # 表示飞行中
            uav.target_station = target_sid
            uav.state = "FLYING"
            #######self.uavs[uav.uid] = uav  # 覆盖
            # 更新站点状态
            station.uav_available.remove(uav.uid) # 从站点移除
            station.orders_waiting = [o for o in station.orders_waiting if o not in orders_to_load]
            ######self.stations[station.sid] = station  # 覆盖



    # ================================================================
    # Rider Action
    # ================================================================
    def _process_rider_action(self, rid, action):
        # 动作: 0=直送, 1..N=去站点
        rider = self.riders[rid]
        
        # 只有当骑手手上有新订单，且还没有确定目标时，动作才生效
        # 或者你允许骑手随时改变主意（Reinforcement Learning通常是每步决策）
        if rider.carrying_order is None:
            return # 空闲骑手无视决策（或用于移动到热点区域，此处简化）

        order = rider.carrying_order
        
        if action == 0:
            # 策略：直接送给客户
            rider.target_pos = order.end
        else:
            # 策略：去站点 (action-1)
            target_sid = action - 1
            if target_sid < self.n_stations:
                rider.target_pos = self.stations[target_sid].pos
            else:
                pass # 无效站点

    # ================================================================
    # 物理移动 & 辅助 
    # ================================================================
    def _move_riders(self):
        for rider in self.riders:
            if rider.target_pos is None:
                continue

            # 1. 检查是否到达 (防止已经在终点但还在计算)
            if np.array_equal(rider.pos, rider.target_pos):
                self._handle_rider_arrival(rider)
                continue

            # 2. 累积移动力
            rider.move_buffer += rider.speed

            # 3. 只有积累满 1.0 移动力，才允许走一格
            if rider.move_buffer >= 1.0:
                rider.move_buffer -= 1.0
                
                # --- 网格寻路逻辑 (自动寻路) ---
                current_pos = rider.pos
                target = rider.target_pos
                diff = target - current_pos # [dx, dy]

                # 策略：朝着距离差距最大的轴走一步 (或者你也可以设为随机选一个轴)
                step = np.array([0, 0], dtype=int)
                
                if abs(diff[0]) > abs(diff[1]):      # X 轴差距更大，优先走 X
                    step[0] = np.sign(diff[0]) # 1 或 -1
                elif abs(diff[1]) > 0:               # Y 轴差距更大，或者 X 轴已经对其了，走 Y
                    step[1] = np.sign(diff[1]) # 1 或 -1

                rider.pos += step

                if np.array_equal(rider.pos, rider.target_pos):
                    self._handle_rider_arrival(rider)
                    rider.move_buffer = 0.0 # 清空 buffer


    def _handle_rider_last_mile(self, station):
        free_riders = [r for r in self.riders if r.carrying_order is None]
        if free_riders:
            rider = random.choice(free_riders)

            order = random.choice(station.orders_to_deliver)
            order.status = ORDER_STATUS["PICKED_BY_R2"]
            order.rider2_id = rider.rid

            station.orders_to_deliver.remove(order)

            rider.pos = np.array(station.pos)
            rider.free = False
            rider.carrying_order = order
            rider.target_pos = order.end
                

                


    def _handle_rider_arrival(self, rider):
        # 骑手到达处理
        if rider.carrying_order:
            order = rider.carrying_order
            # 判定是到了终点还是站点
            is_at_dest = np.array_equal(rider.pos, order.end)
            
            if is_at_dest:
                # 直送完成
                order.status = ORDER_STATUS["DELIVERED"]
                self.active_orders.remove(order)
                rider.carrying_order = None
                rider.target_pos = None # 变为空闲
                rider.free = True
            elif order.status == ORDER_STATUS["PICKED_BY_R1"]:
                # 检查是否到了某个站点
                for station in self.stations:
                    if np.array_equal(rider.pos, station.pos):
                        # 卸货给站点
                        station.orders_waiting.append(order.oid)
                        
                        order.station1_id = station.sid
                        order.status = ORDER_STATUS["AT_STATION"]
                        
                        rider.carrying_order = None
                        rider.target_pos = None
                        rider.free = True
                        break

    def _move_uavs(self):
        for uav in self.uavs:
            if uav.station_id is None: # 飞行中
                target_pos = self.stations[uav.target_station].pos
                vec = target_pos - uav.pos
                dist = np.linalg.norm(vec)
                
                # 耗电
                uav.battery -= 0.001 * uav.speed

                if dist <= uav.speed:
                    # 到达站点
                    uav.pos = target_pos
                    self._handle_uav_arrival(uav)
                else:
                    uav.pos += (vec / dist) * uav.speed

    def _handle_uav_arrival(self, uav):
        # UAV 降落逻辑
        dest_station = self.stations[uav.target_station]
        dest_station.uav_available.append(uav.uid)
        dest_station.orders_to_deliver += uav.orders
        # 1. 卸货
        for order in uav.orders:
            order.status = ORDER_STATUS["AT_DROP_POINT"] 
            order.station2_id = dest_station.sid
        
        uav.orders = []
        
        # 2. 入库
        uav.state = "CHARGING"
        uav.station_id = dest_station.sid
        uav.target_station = None
        
        # 3. 充电
        uav.battery = min(1.0, uav.battery + 0.1)

    def _generate_orders(self):
        if random.random() < 0.2:
            free_riders = [r for r in self.riders if r.carrying_order is None]
            if free_riders:
                rider = random.choice(free_riders)
                # 订单起点也就是随机shop
                start_pos = random.choice(self.shop_locs)
                target_pos = [random.randint(0, self.grid_size), random.randint(0, self.grid_size)]
                order = Order(len(self.orders), start_pos, 
                              target_pos, 
                              self.time)
                order.status = ORDER_STATUS["PICKED_BY_R1"]
                order.rider1_id = rider.rid

                self.orders.append(order)
                self.active_orders.append(order)

                rider.pos = np.array(start_pos)
                rider.free = False
                rider.carrying_order = order
                
                rider.target_pos = None#####################强化学习的决策点##########################
                self.riders[rider.rid] = rider  # 覆盖
                
                # 注意：此时 rider.target_pos 为 None，等待 RL 在下一步 step 给出决策


    # ================================================================
    # ========================  观测函数  =============================
    # ================================================================
    def _get_obs(self, agent):
        # 1. 骑手观测
        if "rider" in agent:
            rid = int(agent.split("_")[1])
            rider = self.riders[rid]
            # ... 返回骑手数据
            return np.zeros(200, dtype=float) # 占位

        # 2. 站点观测 (Station 是决策者)
        elif "station" in agent:
            sid = int(agent.split("_")[1])
            station = self.stations[sid]
            # Station 需要看到：有多少订单在排队？有多少无人机可用？
            # 这里必须返回 station 的状态，而不是 uav 的状态
            return np.zeros(200, dtype=float) # 占位
        
        # 3. 容错
        else:
            return np.zeros(200, dtype=float)

    # ================================================================
    # ========================  奖励函数  =============================
    # ================================================================
    def _compute_reward(self, agent):
        reward = -1  # 每步 -1

        # 完成订单奖励
        for order in self.active_orders:
            if order.status == ORDER_STATUS["DELIVERED"]:
                reward += 50

            # 延迟惩罚
            reward -= 0.01 * order.time_wait

            # 超时惩罚
            if order.time_wait > 60:
                reward -= 100

        return reward
