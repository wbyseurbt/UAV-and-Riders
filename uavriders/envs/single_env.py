from __future__ import annotations

import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces

from uavriders.configs.env_config import EnvConfig, default_config
from uavriders.sim.entities import Rider, Station, UAV
from uavriders.rl.observations import get_obs
from uavriders.sim.order_flow import add_manual_order as add_manual_order_impl
from uavriders.sim.order_flow import generate_orders
from uavriders.rl.rewards import compute_reward_components
from uavriders.sim.rider_logic import handle_rider_last_mile, move_riders, process_rider_action
from uavriders.sim.station_logic import process_station_action
from uavriders.envs.state import EnvData
from uavriders.sim.uav_logic import charge_uavs, move_uavs


class DeliveryUAVSingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 200, seed: int | None = None, config: EnvConfig | dict | None = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        if config is None:
            self.cfg = default_config()
        elif isinstance(config, EnvConfig):
            self.cfg = config
        else:
            self.cfg = EnvConfig.from_dict(config)

        self.data = EnvData()

        n_stations = int(self.cfg.n_stations)
        n_riders = int(self.cfg.n_riders)

        self._rider_obs_dim = 6 + (n_stations * 3)
        self._station_obs_dim = 6 + (n_stations * 2)##特征维度

        self._subaction_n = int(1 + n_stations)
        self._action_len = int(n_riders + n_stations)
        self.action_space = spaces.MultiDiscrete([self._subaction_n] * self._action_len)

        self._obs_len = int(n_riders * self._rider_obs_dim + n_stations * self._station_obs_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_len,), dtype=np.float32)

        self._delivered_this_step = 0
        self._uav_launch_this_step = 0
        self._uav_order_balance = 0
        self.force_station_prob = float(self.cfg.force_station_prob)
        self._handoff_this_step = 0
        self._handoff_optimal_this_step = 0

    @property
    def time(self) -> int:
        return int(self.data.time)

    @time.setter
    def time(self, value: int):
        self.data.time = int(value)

    @property
    def orders(self):
        return self.data.orders

    @orders.setter
    def orders(self, value):
        self.data.orders = value

    @property
    def active_orders(self):
        return self.data.active_orders

    @active_orders.setter
    def active_orders(self, value):
        self.data.active_orders = value

    @property
    def stations(self):
        return self.data.stations

    @stations.setter
    def stations(self, value):
        self.data.stations = value

    @property
    def uavs(self):
        return self.data.uavs

    @uavs.setter
    def uavs(self, value):
        self.data.uavs = value

    @property
    def riders(self):
        return self.data.riders

    @riders.setter
    def riders(self, value):
        self.data.riders = value

    @property
    def grid_size(self) -> int:
        return int(self.cfg.world_grid_size)

    @property
    def n_riders(self) -> int:
        return int(self.cfg.n_riders)

    @property
    def n_stations(self) -> int:
        return int(self.cfg.n_stations)

    @property
    def n_uavs(self) -> int:
        return int(self.cfg.n_uavs)

    @property
    def n_shops(self) -> int:
        return int(self.cfg.n_shops)

    @property
    def station_locs(self):
        return [np.array(pos, dtype=int) for pos in self.cfg.station_locs]

    @property
    def shop_locs(self):
        return [np.array(pos, dtype=int) for pos in self.cfg.shop_locs]

    def _pack_obs(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for rid in range(self.n_riders):
            parts.append(np.asarray(get_obs(self, f"rider_{rid}"), dtype=np.float32).reshape(-1))
        for sid in range(self.n_stations):
            parts.append(np.asarray(get_obs(self, f"station_{sid}"), dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _split_action(self, action) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.shape[0] != self._action_len:
            raise ValueError(f"action length mismatch: got {arr.shape[0]}, expected {self._action_len}")
        rider_actions = arr[: self.n_riders]
        station_actions = arr[self.n_riders :]
        return rider_actions, station_actions

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng.seed(seed)

        self.data = EnvData()
        self._delivered_this_step = 0

        self.stations = [Station(i, pos, self.cfg) for i, pos in enumerate(self.station_locs)]

        self.uavs = []
        for uid in range(self.n_uavs):
            sid = uid % self.n_stations
            u = UAV(uid, self.stations[sid].pos, sid, self.cfg)
            self.uavs.append(u)
            self.stations[sid].uav_available.append(uid)

        self.riders = []
        for rid in range(self.n_riders):
            x = self._rng.randint(0, self.grid_size)
            y = self._rng.randint(0, self.grid_size)
            self.riders.append(Rider(rid, (x, y), self.cfg))

        obs = self._pack_obs()
        return obs, {"infos": {}}

    def step(self, action):
        self.time = self.time + 1
        self._delivered_this_step = 0
        self._uav_launch_this_step = 0
        self._uav_order_balance = 0
        self._handoff_this_step = 0
        self._handoff_optimal_this_step = 0

        for o in self.active_orders:
            o.time_wait += 1

        generate_orders(self)

        rider_actions, station_actions = self._split_action(action)

        for sid in range(self.n_stations):
            process_station_action(self, sid, int(station_actions[sid]))

        for rid in range(self.n_riders):
            process_rider_action(self, rid, int(rider_actions[rid]))

        move_riders(self)
        move_uavs(self)
        charge_uavs(self)

        for station in self.stations:
            handle_rider_last_mile(self, station)

        obs = self._pack_obs()

        reward_components = compute_reward_components(self)
        reward = float(sum(reward_components.values()))
        terminated = self.time >= self.max_steps
        truncated = False
        if len(self.active_orders) > 20:
            truncated = True
            reward = -200.0
            reward_components = {"overflow_truncate_penalty": -200.0}
        return obs, float(reward), bool(terminated), bool(truncated), {"reward_components": reward_components}

    def set_force_station_prob(self, prob: float):
        self.force_station_prob = float(prob)

    def add_manual_order(self, start_pos, end_pos):
        add_manual_order_impl(self, start_pos, end_pos)

    def close(self):
        return
