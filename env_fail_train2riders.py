import numpy as np
import random
from pettingzoo import ParallelEnv
from gymnasium import spaces
import env_config as cfg

# ================================================================
# Order status (must follow the required sequence)
# ================================================================
ORDER_STATUS = {
    "UNASSIGNED":       0,
    "PICKED_BY_R1":     1,  # Rider1 has accepted & picked up (restaurant pickup simplified)
    "AT_STATION":       2,  # Arrived at UAV station, waiting for UAV
    "IN_UAV":           3,  # In UAV transport
    "AT_DROP_POINT":    4,  # Arrived at destination station (UAV drop)
    "PICKED_BY_R2":     5,  # Rider2 picked up for last mile
    "DELIVERED":        6,  # Delivered
}

# ================================================================
# Entities
# ================================================================
class Order:
    def __init__(self, oid, start, end, time_created):
        self.oid = int(oid)
        self.start = np.array(start, dtype=int)   # shop (restaurant)
        self.end = np.array(end, dtype=int)       # customer
        self.time_created = int(time_created)
        self.time_wait = 0                        # waiting time in system since creation

        self.status = ORDER_STATUS["UNASSIGNED"]

        self.rider1_id = None
        self.station1_id = None
        self.uav_id = None
        self.station2_id = None
        self.rider2_id = None


class Rider:
    def __init__(self, rid, pos):
        self.rid = int(rid)
        self.pos = np.array(pos, dtype=int)  # grid integer
        self.free = True
        self.carrying_order = None  # Order object not id
        self.target_pos = None      # grid integer target

        # speed: 0.5 grid/min -> buffer trick
        self.move_buffer = 0.0
        self.speed = float(cfg.Rider_SPEED)

    def reset(self, pos):
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None
        self.move_buffer = 0.0


class UAV:
    def __init__(self, uid, pos, station_id):
        self.uid = int(uid)
        self.pos = np.array(pos, dtype=float)   # continuous
        self.station_id = int(station_id)       # docked station id; None means flying
        self.target_station = None              # station id when flying

        self.speed = float(cfg.UAV_SPEED)
        self.capacity_limit = int(cfg.UAV_capacity_limit)

        self.battery = 1.0
        self.state = "STOP"  # STOP / FLYING / CHARGING

        # IMPORTANT: store ORDER IDs only
        self.orders = []

    @property
    def is_full(self):
        return len(self.orders) >= self.capacity_limit


class Station:
    def __init__(self, sid, pos):
        self.sid = int(sid)
        self.pos = np.array(pos, dtype=int)

        self.max_uavs = int(cfg.Station_MAX_UAVS)
        self.max_order_buffer = int(cfg.Station_MAX_ORDER_BUFFER)

        # IMPORTANT: store IDs only
        self.uav_available = []     # list[int] global uav ids currently docked & available
        self.orders_waiting = []    # list[int] order ids waiting for UAV pickup at this station
        self.orders_to_deliver = [] # list[int] order ids waiting for Rider2 pickup at this station


# ================================================================
# Environment
# ================================================================
class DeliveryUAVEnv(ParallelEnv):
    """
    Trainable PettingZoo ParallelEnv aligned with the paper:
    - Riders (agents) decide: direct-to-destination vs go-to-station(k)
    - Stations (agents) decide: for each launch slot, pick 1 UAV + target station + up to Cap orders
    - UAVs are executors (not agents)
    """
    metadata = {"name": "delivery_uav_trainable_v1"}

    def __init__(self, max_steps:int = 200, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        # world params
        self.grid_size = int(cfg.World_grid_size)
        self.n_riders = int(cfg.World_n_riders)
        self.n_stations = int(cfg.World_n_stations)
        self.n_uavs = int(cfg.World_n_uavs)
        self.n_shops = int(cfg.World_n_shops)

        self.station_locs = [np.array(pos, dtype=int) for pos in cfg.World_locs_stations]
        self.shop_locs = [np.array(pos, dtype=int) for pos in cfg.World_locs_shops]

        # station action limitation (fixed action space)
        self.concurrent_launches = int(cfg.Station_MAX_CONCURRENT_LAUNCH)
        self.uav_cap = int(cfg.UAV_capacity_limit)

        # Agent names
        self.rider_agents = [f"rider_{i}" for i in range(self.n_riders)]
        self.station_agents = [f"station_{k}" for k in range(self.n_stations)]
        self.agents = self.rider_agents + self.station_agents

        # ---------------- Action spaces ----------------
        self.action_spaces = {}

        # Rider: 0=direct, 1..N=go station (sid=action-1)
        for a in self.rider_agents:
            self.action_spaces[a] = spaces.Discrete(1 + self.n_stations)

        # Station: fixed MultiDiscrete via LOCAL indices
        # single launch slot dims:
        #  [uav_pick, target_station, order_pick_1, ..., order_pick_cap]
        # uav_pick: 0=noop, 1..Station_MAX_UAVS means pick that index in station.uav_available
        # order_pick: 0=empty, 1..Station_MAX_ORDER_BUFFER means pick that index in station.orders_waiting
        max_uav_pick = int(cfg.Station_MAX_UAVS) + 1
        max_order_pick = int(cfg.Station_MAX_ORDER_BUFFER) + 1
        target_station_dim = self.n_stations

        single_launch_dims = [max_uav_pick, target_station_dim] + [max_order_pick] * self.uav_cap
        self.single_action_len = len(single_launch_dims)
        full_station_dims = single_launch_dims * self.concurrent_launches

        for a in self.station_agents:
            self.action_spaces[a] = spaces.MultiDiscrete(full_station_dims)

        # ---------------- Observation spaces ----------------
        # Keep them compact & fixed-size for training
        # Rider obs (float32, shape=(10,)):
        # [x,y, has_order, order_wait_norm, dist_to_dest_norm, dist_to_nearest_station_norm,
        #  time_norm, dummy1, dummy2, dummy3]
        self._rider_obs_dim = 10

        # Station obs (float32, shape=(12,)):
        # [x,y, num_uav_avail_norm, num_waiting_norm, num_to_deliver_norm,
        #  avg_battery_avail, min_wait_norm, time_norm, dummy...]
        self._station_obs_dim = 12

        self.observation_spaces = {}
        for a in self.rider_agents:
            self.observation_spaces[a] = spaces.Box(low=-1.0, high=1.0, shape=(self._rider_obs_dim,), dtype=np.float32)
        for a in self.station_agents:
            self.observation_spaces[a] = spaces.Box(low=-1.0, high=1.0, shape=(self._station_obs_dim,), dtype=np.float32)

        # runtime containers
        self.time = 0
        self.orders: list[Order] = []
        self.active_orders: list[Order] = []
        self.stations: list[Station] = []
        self.uavs: list[UAV] = []
        self.riders: list[Rider] = []

        # bookkeeping for reward
        self._delivered_this_step = 0

        # UAV launch cost bookkeeping
        self._uav_launch_this_step = 0

    # RLlib convenience
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ---------------- PettingZoo API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        self.time = 0
        self.orders = []
        self.active_orders = []
        self._delivered_this_step = 0

        # init stations
        self.stations = [Station(i, pos) for i, pos in enumerate(self.station_locs)]

        # init uavs, distribute to stations
        self.uavs = []
        for uid in range(self.n_uavs):
            sid = uid % self.n_stations
            u = UAV(uid, self.stations[sid].pos, sid)
            self.uavs.append(u)
            self.stations[sid].uav_available.append(uid)

        # init riders at random grid positions
        self.riders = []
        for rid in range(self.n_riders):
            x = self._rng.randint(0, self.grid_size)
            y = self._rng.randint(0, self.grid_size)
            self.riders.append(Rider(rid, (x, y)))

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self.time += 1
        self._delivered_this_step = 0
        self._uav_launch_this_step = 0

        # update waiting time
        for o in self.active_orders:
            o.time_wait += 1

        # new orders
        self._generate_orders()

        # stations act first (launch UAVs)
        for agent_name, act in actions.items():
            if agent_name.startswith("station_"):
                sid = int(agent_name.split("_")[1])
                self._process_station_action(sid, act)

        # riders act (set their targets for current carried orders)
        for agent_name, act in actions.items():
            if agent_name.startswith("rider_"):
                rid = int(agent_name.split("_")[1])
                self._process_rider_action(rid, int(act))

        # physics
        self._move_riders()
        self._move_uavs()

        # charge UAVs
        self._charge_uavs()

        # after UAV arrivals, assign last-mile riders if possible
        for station in self.stations:
            self._handle_rider_last_mile(station)

        # observations
        obs = {a: self._get_obs(a) for a in self.agents}

        # rewards (shared global reward; MAPPO can still learn with shared signals)
        shared_reward = self._compute_shared_reward()
        rewards = {a: float(shared_reward) for a in self.agents}

        done = self.time >= self.max_steps

        terminated = {agent: done for agent in self.agents}
        terminated["__all__"] = done

        truncated = {agent: False for agent in self.agents}
        truncated["__all__"] = False

        infos = {a: {} for a in self.agents}

        return obs, rewards, terminated, truncated, infos

    # ================================================================
    # Station action: batch launch processing (LOCAL indices -> GLOBAL ids)
    # ================================================================
    def _process_station_action(self, sid, action_vec):
        station = self.stations[sid]

        used_uavs = set()
        used_orders = set()

        action_vec = np.asarray(action_vec, dtype=int).tolist()

        for k in range(self.concurrent_launches):
            start = k * self.single_action_len
            end = (k + 1) * self.single_action_len
            sub = action_vec[start:end]

            uav_pick = int(sub[0])        # 0=noop
            target_sid = int(sub[1])      # 0..n_stations-1
            order_picks = [int(x) for x in sub[2:]]

            if uav_pick == 0:
                continue
            if target_sid < 0 or target_sid >= self.n_stations:
                continue

            # map local UAV index -> global UAV id
            local_uav_idx = uav_pick - 1
            if local_uav_idx < 0 or local_uav_idx >= len(station.uav_available):
                continue
            uav_id = station.uav_available[local_uav_idx]

            if uav_id in used_uavs:
                continue
            uav = self.uavs[uav_id]

            # battery constraint (paper uses 0.1 bmax)
            if uav.battery < 0.1:
                continue

            # collect orders to load (map local indices -> global order ids)
            oids_to_load = []
            for op in order_picks:
                if op == 0:
                    continue
                local_order_idx = op - 1
                if local_order_idx < 0 or local_order_idx >= len(station.orders_waiting):
                    continue
                oid = station.orders_waiting[local_order_idx]
                if oid in used_orders:
                    continue
                oids_to_load.append(oid)
                if len(oids_to_load) >= uav.capacity_limit:
                    break

            # if not oids_to_load:
            #     continue

            # mark used
            used_uavs.add(uav_id)
            for oid in oids_to_load:
                used_orders.add(oid)

            # update orders
            if oids_to_load:
                for oid in oids_to_load:
                    o = self.orders[oid]
                    o.status = ORDER_STATUS["IN_UAV"]
                    o.uav_id = uav_id

            # configure UAV
            uav.orders = list(oids_to_load)
            uav.station_id = None
            uav.target_station = target_sid
            uav.state = "FLYING"

            # UAV 起飞计数（用于奖励函数中的成本项）
            self._uav_launch_this_step += 1

            # update station buffers
            station.uav_available.remove(uav_id)
            # remove loaded orders (by id)
            station.orders_waiting = [x for x in station.orders_waiting if x not in oids_to_load]

    # ================================================================
    # Rider action: set target for carried order
    # ================================================================
    def _process_rider_action(self, rid, action):
        rider = self.riders[rid]
        if rider.carrying_order is None:
            # simple behavior: move towards nearest station if action > 0
            if action > 0:
                sid = action - 1
                if 0 <= sid < self.n_stations:
                    rider.target_pos = self.stations[sid].pos.copy()
            return

        order = rider.carrying_order

        is_last_mile = (order.status == ORDER_STATUS["PICKED_BY_R2"])

        # 如果是最后一公里，强制无视 AI 的去站点指令，强制设为直送 (Action 0)
        if is_last_mile or action == 0:
            rider.target_pos = order.end.copy()
        else:
            # 只有在第一阶段 (PICKED_BY_R1)，才允许去中转站
            target_sid = action - 1
            if 0 <= target_sid < self.n_stations:
                rider.target_pos = self.stations[target_sid].pos.copy()

    # ================================================================
    # Physics: riders
    # ================================================================
    def _move_riders(self):
        for rider in self.riders:
            if rider.target_pos is None:
                continue

            # already there
            if np.array_equal(rider.pos, rider.target_pos):
                self._handle_rider_arrival(rider)
                continue

            rider.move_buffer += rider.speed

            if rider.move_buffer >= 1.0:
                rider.move_buffer -= 1.0

                cur = rider.pos
                tgt = rider.target_pos
                diff = tgt - cur

                step = np.array([0, 0], dtype=int)
                if abs(diff[0]) > abs(diff[1]):
                    step[0] = int(np.sign(diff[0]))
                elif abs(diff[1]) > 0:
                    step[1] = int(np.sign(diff[1]))
                rider.pos = rider.pos + step

                if np.array_equal(rider.pos, rider.target_pos):
                    self._handle_rider_arrival(rider)
                    rider.move_buffer = 0.0

    def _handle_rider_arrival(self, rider: Rider):
        if rider.carrying_order is None:
            rider.target_pos = None
            rider.free = True
            return

        o = rider.carrying_order

        # Case 1: arrived at customer destination -> delivered
        if np.array_equal(rider.pos, o.end):
            o.status = ORDER_STATUS["DELIVERED"]
            self._delivered_this_step += 1
            if o in self.active_orders:
                self.active_orders.remove(o)

            rider.carrying_order = None
            rider.target_pos = None
            rider.free = True
            return

        # Case 2: rider1 arrives station to handoff
        if o.status == ORDER_STATUS["PICKED_BY_R1"]:
            for st in self.stations:
                if np.array_equal(rider.pos, st.pos):
                    # put order into station waiting list
                    if o.oid not in st.orders_waiting:
                        st.orders_waiting.append(o.oid)

                    o.station1_id = st.sid
                    o.status = ORDER_STATUS["AT_STATION"]

                    rider.carrying_order = None
                    rider.target_pos = None
                    rider.free = True
                    return
                
        if rider.carrying_order is not None and rider.target_pos is None:
        # 如果到了地方却没法处理（既不是客户家，也不是R1交接），说明走错路了
        # 强制让他继续去客户家
            rider.target_pos = rider.carrying_order.end.copy()
            return 

        # # Case 3: rider2 arrives station? (not needed in this simplified last-mile assignment)
        # # If not matched, keep idle
        # rider.target_pos = None



    # ================================================================
    # Physics: UAVs
    # ================================================================
    def _move_uavs(self):
        for uav in self.uavs:
            if uav.station_id is None and uav.target_station is not None:
                target_pos = self.stations[uav.target_station].pos.astype(float)
                vec = target_pos - uav.pos
                dist = float(np.linalg.norm(vec) + 1e-9)

                # battery consumption (simple)
                uav.battery = max(0.0, uav.battery - 0.001 * uav.speed)

                if dist <= uav.speed:
                    uav.pos = target_pos
                    self._handle_uav_arrival(uav)
                else:
                    uav.pos = uav.pos + (vec / dist) * uav.speed

    def _handle_uav_arrival(self, uav: UAV):
        dest_station = self.stations[uav.target_station]

        # capacity constraint at hub (if over, still land but penalize via reward)
        dest_station.uav_available.append(uav.uid)

        # unload: uav.orders are order ids
        dest_station.orders_to_deliver.extend(uav.orders)

        for oid in uav.orders:
            o = self.orders[oid]
            o.status = ORDER_STATUS["AT_DROP_POINT"]
            o.station2_id = dest_station.sid

        # reset uav
        uav.orders = []
        uav.state = "CHARGING"
        uav.station_id = dest_station.sid
        uav.target_station = None

        # charge
        uav.battery = min(1.0, uav.battery + 0.1)

    # ================================================================
    # Last mile: assign Rider2 (kept simple; can be learned later)
    # ================================================================
    def _handle_rider_last_mile(self, station: Station):
        if not station.orders_to_deliver:
            return

        free_riders = [r for r in self.riders if r.carrying_order is None]
        if not free_riders:
            return

        # Assign at most one per step per station (simple)
        rider = self._rng.choice(free_riders)
        oid = self._rng.choice(station.orders_to_deliver)
        station.orders_to_deliver.remove(oid)

        o = self.orders[oid]
        o.status = ORDER_STATUS["PICKED_BY_R2"]
        o.rider2_id = rider.rid

        rider.pos = station.pos.copy()
        rider.free = False
        rider.carrying_order = o
        rider.target_pos = o.end.copy()


    # ===============================================================
    # Battery charging
    #================================================================
    def _charge_uavs(self):
        for uav in self.uavs:
            if uav.state == "CHARGING":
                uav.battery = min(1.0, uav.battery + 0.1)
                if uav.battery >= 1.0:
                    uav.state = "STOP"


    # ================================================================
    # Order generation: simplified (restaurant pickup ignored)
    # ================================================================
    def _generate_orders(self):
        # arrival probability (scaled)
        if self._rng.random() < 0.2:
            free_riders = [r for r in self.riders if r.carrying_order is None]
            if not free_riders:
                return

            rider = self._rng.choice(free_riders)

            start_pos = self._rng.choice(self.shop_locs).copy()
            end_pos = np.array([self._rng.randint(0, self.grid_size), self._rng.randint(0, self.grid_size)], dtype=int)

            oid = len(self.orders)
            o = Order(oid, start_pos, end_pos, self.time)
            o.status = ORDER_STATUS["PICKED_BY_R1"]
            o.rider1_id = rider.rid

            self.orders.append(o)
            self.active_orders.append(o)

            # place rider at shop instantly (pickup simplified)
            rider.pos = start_pos.copy()
            rider.free = False
            rider.carrying_order = o
            rider.target_pos = None  # DECISION POINT (RL sets next target)

    # ================================================================
    # Observations (normalized to [-1, 1] roughly)
    # ================================================================
    def _get_obs(self, agent):
        if agent.startswith("rider_"):
            rid = int(agent.split("_")[1])
            r = self.riders[rid]

            x = r.pos[0] / max(1, self.grid_size)
            y = r.pos[1] / max(1, self.grid_size)

            has_order = 1.0 if r.carrying_order is not None else 0.0
            if r.carrying_order is not None:
                o = r.carrying_order
                wait_norm = min(1.0, o.time_wait / 60.0)
                dist_to_dest = self._manhattan(r.pos, o.end) / (2 * max(1, self.grid_size))
            else:
                wait_norm = 0.0
                dist_to_dest = 0.0

            # nearest station distance
            d_near = min(self._manhattan(r.pos, st.pos) for st in self.stations) / (2 * max(1, self.grid_size))
            
            nearest_st = min(self.stations, key=lambda s: self._manhattan(r.pos, s.pos))
            # 最近站点的拥堵程度
            # 如果那个站点排队很长，这个值会很大 (接近1.0)
            # 骑手 AI 看到这个值很高，就会学到：“太堵了，我还是自己送吧”
            st_congestion = min(1.0, len(nearest_st.orders_waiting) / max(1, cfg.Station_MAX_ORDER_BUFFER))

            time_norm = min(1.0, self.time / max(1, self.max_steps))

            obs = np.array([x, y, has_order, wait_norm, dist_to_dest, d_near, time_norm, st_congestion, 0.0, 0.0], dtype=np.float32)
            return obs

        if agent.startswith("station_"):
            sid = int(agent.split("_")[1])
            st = self.stations[sid]

            x = st.pos[0] / max(1, self.grid_size)
            y = st.pos[1] / max(1, self.grid_size)

            uav_avail = min(1.0, len(st.uav_available) / max(1, cfg.Station_MAX_UAVS))
            waiting = min(1.0, len(st.orders_waiting) / max(1, cfg.Station_MAX_ORDER_BUFFER))
            to_deliver = min(1.0, len(st.orders_to_deliver) / max(1, cfg.Station_MAX_ORDER_BUFFER))

            if st.uav_available:
                bats = [self.uavs[uid].battery for uid in st.uav_available]
                avg_bat = float(np.mean(bats))
            else:
                avg_bat = 0.0

            if st.orders_waiting:
                waits = [self.orders[oid].time_wait for oid in st.orders_waiting]
                max_wait = min(1.0, float(max(waits)) / 60.0)
            else:
                max_wait = 0.0

            time_norm = min(1.0, self.time / max(1, self.max_steps))

            obs = np.array(
                [x, y, uav_avail, waiting, to_deliver, avg_bat, max_wait, time_norm, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32
            )
            return obs

        # fallback
        return np.zeros((12,), dtype=np.float32)

    # ================================================================
    # Reward (shared) aligned with paper components (simplified)
    # ================================================================
    def _compute_shared_reward(self) -> float:
        # per-step penalty per active order
        r = -0.01 * len(self.active_orders)

        # completion reward (count deliveries this step)
        r += 0.5 * self._delivered_this_step

        # delay penalty
        total_wait = sum(o.time_wait for o in self.active_orders)
        r -= 0.0001 * float(total_wait)

        # overtime penalty (>60 min)
        overtime = sum(1 for o in self.active_orders if o.time_wait > 60)
        r -= 1.0 * float(overtime)

        # UAV fly cost
        UAV_LAUNCH_COST = 0.0001  # 可调超参数：每次起飞的成本
        r -= UAV_LAUNCH_COST * float(self._uav_launch_this_step)

        # hub overflow penalty
        overflow = 0
        for st in self.stations:
            if len(st.uav_available) > st.max_uavs:
                overflow += (len(st.uav_available) - st.max_uavs)
        if overflow > 0:
            r -= 0.02 * float(overflow)

        return float(r)

    @staticmethod
    def _manhattan(p, q) -> int:
        p = np.asarray(p, dtype=int)
        q = np.asarray(q, dtype=int)
        return int(abs(p[0] - q[0]) + abs(p[1] - q[1]))





    # ================================================================
    # [新增] 手动插入订单接口
    # ================================================================
    def add_manual_order(self, start_pos, end_pos):
        """
        允许外部 UI 直接插入一个订单
        start_pos: (x, y) 商店位置
        end_pos: (x, y) 顾客位置
        """
        # 寻找最近的空闲骑手 (为了简化逻辑，模拟 R1 取餐)
        free_riders = [r for r in self.riders if r.carrying_order is None]
        
        # 如果没有空闲骑手，强制让第一个骑手接单 (或者你可以选择忽略)
        # 这里为了演示效果，我们强制指派，即使这可能打断他原来的逻辑(严谨点应该只选free的)
        if free_riders:
            rider = min(free_riders, key=lambda r: self._manhattan(r.pos, start_pos))
        else:
            # 如果大家都在忙，就暂时不生成，或者随机选一个忙碌的重置（不推荐），
            # 这里简单处理：如果没有空闲骑手，打印提示并跳过
            print("无法手动下单：没有空闲骑手！")
            return

        oid = len(self.orders)
        o = Order(oid, start_pos, end_pos, self.time)
        o.status = ORDER_STATUS["PICKED_BY_R1"]
        o.rider1_id = rider.rid

        self.orders.append(o)
        self.active_orders.append(o)

        # 瞬移骑手到商店并接单 (简化模拟)
        rider.pos = np.array(start_pos, dtype=int)
        rider.free = False
        rider.carrying_order = o
        rider.target_pos = None 
        
        print(f"手动订单已生成: Order {oid} | {start_pos} -> {end_pos} | Rider {rider.rid}")