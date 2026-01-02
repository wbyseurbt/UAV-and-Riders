import numpy as np
import random
from pettingzoo import ParallelEnv
from gymnasium import spaces
import env_config as cfg  # 确保 env_config.py 在同一目录下

# ================================================================
# 基础实体定义 (为了独立运行，这里重新定义一遍简化版)
# ================================================================
ORDER_STATUS = {
    "UNASSIGNED":       0,
    "PICKED_BY_RIDER":  1,
    "DELIVERED":        2,
}

class Order:
    def __init__(self, oid, start, end, time_created):
        self.oid = int(oid)
        self.start = np.array(start, dtype=int)
        self.end = np.array(end, dtype=int)
        self.time_created = int(time_created)
        self.time_wait = 0
        self.status = ORDER_STATUS["UNASSIGNED"]

class Rider:
    def __init__(self, rid, pos):
        self.rid = int(rid)
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None
        self.move_buffer = 0.0
        self.speed = float(cfg.Rider_SPEED) # 读取配置

    def reset(self, pos):
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None
        self.move_buffer = 0.0

# ================================================================
# 纯骑手基准环境 (Pure Rider Environment)
# ================================================================
class PureRiderEnv(ParallelEnv):
    metadata = {"name": "pure_rider_baseline_v1"}

    def __init__(self, max_steps: int = 300, seed: int | None = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        # 读取世界配置
        self.grid_size = int(cfg.World_grid_size)
        self.n_riders = int(cfg.World_n_riders)
        self.n_shops = int(cfg.World_n_shops)
        self.shop_locs = [np.array(pos, dtype=int) for pos in cfg.World_locs_shops]

        # 代理名称 (只有骑手)
        self.agents = [f"rider_{i}" for i in range(self.n_riders)]
        self.possible_agents = self.agents[:]

        # 动作空间: Discrete(1) -> 实际上就是无操作，因为路径是强制锁定的
        self.action_spaces = {a: spaces.Discrete(1) for a in self.agents}

        # 观察空间: 保持和原环境一致的维度 (10,) 方便神经网络处理(虽然这里不需要神经网络)
        self._rider_obs_dim = 10
        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(self._rider_obs_dim,), dtype=np.float32)
            for a in self.agents
        }

        # 运行时变量
        self.time = 0
        self.orders = []
        self.active_orders = []
        self.riders = []
        self._delivered_this_step = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        self.time = 0
        self.orders = []
        self.active_orders = []
        self._delivered_this_step = 0
        self.agents = self.possible_agents[:]

        # 初始化骑手位置
        self.riders = []
        for rid in range(self.n_riders):
            x = self._rng.randint(0, self.grid_size)
            y = self._rng.randint(0, self.grid_size)
            self.riders.append(Rider(rid, (x, y)))

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        """
        注意：这里的 actions 参数会被忽略，因为纯骑手模式下，
        骑手的行为是规则固定的 (Rule-based)：有单就去送，没单就停。
        """
        self.time += 1
        self._delivered_this_step = 0

        # 更新等待时间
        for o in self.active_orders:
            o.time_wait += 1

        # 生成订单
        self._generate_orders()

        # 核心逻辑：强制设置骑手目标
        for rider in self.riders:
            self._update_rider_target(rider)

        # 物理移动
        self._move_riders()

        # 获取状态和奖励
        obs = {a: self._get_obs(a) for a in self.agents}
        
        # 计算奖励 (使用与训练环境相同的逻辑以便对比)
        shared_reward = self._compute_shared_reward()
        rewards = {a: float(shared_reward) for a in self.agents}

        done = self.time >= self.max_steps
        terminated = {a: done for a in self.agents}
        terminated["__all__"] = done
        truncated = {a: False for a in self.agents}
        truncated["__all__"] = False
        infos = {a: {} for a in self.agents}

        return obs, rewards, terminated, truncated, infos

    # --- 内部逻辑 ---

    def _update_rider_target(self, rider: Rider):
        # 如果没带货，就没有目标 (原地待命)
        if rider.carrying_order is None:
            rider.target_pos = None
            return
        
        # 如果带货了，目标强制锁定为客户位置
        rider.target_pos = rider.carrying_order.end.copy()

    def _move_riders(self):
        for rider in self.riders:
            if rider.target_pos is None:
                continue

            # 到达检测
            if np.array_equal(rider.pos, rider.target_pos):
                self._handle_arrival(rider)
                continue

            # 移动逻辑
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
                    self._handle_arrival(rider)
                    rider.move_buffer = 0.0

    def _handle_arrival(self, rider):
        if rider.carrying_order is None:
            return
        
        o = rider.carrying_order
        if np.array_equal(rider.pos, o.end):
            o.status = ORDER_STATUS["DELIVERED"]
            self._delivered_this_step += 1
            if o in self.active_orders:
                self.active_orders.remove(o)
            
            rider.carrying_order = None
            rider.target_pos = None
            rider.free = True

    def _generate_orders(self):
        if self._rng.random() < 0.2:
            free_riders = [r for r in self.riders if r.carrying_order is None]
            if not free_riders:
                return

            rider = self._rng.choice(free_riders)
            start_pos = self._rng.choice(self.shop_locs).copy()
            end_pos = np.array([self._rng.randint(0, self.grid_size), self._rng.randint(0, self.grid_size)], dtype=int)

            oid = len(self.orders)
            o = Order(oid, start_pos, end_pos, self.time)
            o.status = ORDER_STATUS["PICKED_BY_RIDER"]

            self.orders.append(o)
            self.active_orders.append(o)

            # 瞬间取货 (和原环境保持一致假设)
            rider.pos = start_pos.copy()
            rider.free = False
            rider.carrying_order = o
            rider.target_pos = end_pos 

    def _get_obs(self, agent_id):
        rid = int(agent_id.split("_")[1])
        r = self.riders[rid]

        x = r.pos[0] / max(1, self.grid_size)
        y = r.pos[1] / max(1, self.grid_size)

        has_order = 1.0 if r.carrying_order is not None else 0.0
        dist_to_dest = 0.0
        wait_norm = 0.0

        if r.carrying_order:
            o = r.carrying_order
            wait_norm = min(1.0, o.time_wait / 60.0)
            dist_to_dest = self._manhattan(r.pos, o.end) / (2 * max(1, self.grid_size))

        d_near = 1.0 
        time_norm = min(1.0, self.time / max(1, self.max_steps))

        return np.array([x, y, has_order, wait_norm, dist_to_dest, d_near, time_norm, 0.0, 0.0, 0.0], dtype=np.float32)

    def _compute_shared_reward(self) -> float:
        # 只保留和订单相关的奖励，去掉了中转站溢出的惩罚
        r = -0.01 * len(self.active_orders)
        r += 0.5 * self._delivered_this_step
        total_wait = sum(o.time_wait for o in self.active_orders)
        r -= 0.0001 * float(total_wait)
        overtime = sum(1 for o in self.active_orders if o.time_wait > 60)
        r -= 1.0 * float(overtime)
        return float(r)

    @staticmethod
    def _manhattan(p, q) -> int:
        p = np.asarray(p, dtype=int)
        q = np.asarray(q, dtype=int)
        return int(abs(p[0] - q[0]) + abs(p[1] - q[1]))