from __future__ import annotations

import numpy as np

from uavriders.configs.env_config import EnvConfig


ORDER_STATUS: dict[str, int] = {
    "UNASSIGNED": 0,
    "PICKED_BY_R1": 1,
    "AT_STATION": 2,
    "IN_UAV": 3,
    "AT_DROP_POINT": 4,
    "PICKED_BY_R2": 5,
    "DELIVERED": 6,
}


class Order:
    def __init__(self, oid, start, end, time_created):
        self.oid = int(oid)
        self.start = np.array(start, dtype=int)
        self.end = np.array(end, dtype=int)
        self.time_created = int(time_created)
        self.time_wait = 0

        self.status = ORDER_STATUS["UNASSIGNED"]

        self.rider1_id = None
        self.station1_id = None
        self.uav_id = None
        self.station2_id = None
        self.rider2_id = None


class Rider:
    def __init__(self, rid, pos, cfg: EnvConfig):
        self.rid = int(rid)
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None

        self.move_buffer = 0.0
        self.speed = float(cfg.rider_speed)

    def reset(self, pos):
        self.pos = np.array(pos, dtype=int)
        self.free = True
        self.carrying_order = None
        self.target_pos = None
        self.move_buffer = 0.0


class UAV:
    def __init__(self, uid, pos, station_id, cfg: EnvConfig):
        self.uid = int(uid)
        self.pos = np.array(pos, dtype=float)
        self.station_id = int(station_id)
        self.target_station = None

        self.speed = float(cfg.uav_speed)
        self.capacity_limit = int(cfg.uav_capacity_limit)

        self.battery = 1.0
        self.state = "STOP"

        self.orders = []

    @property
    def is_full(self):
        return len(self.orders) >= self.capacity_limit


class Station:
    def __init__(self, sid, pos, cfg: EnvConfig):
        self.sid = int(sid)
        self.pos = np.array(pos, dtype=int)

        self.max_uavs = int(cfg.station_max_uavs)
        self.max_order_buffer = int(cfg.station_max_order_buffer)

        self.uav_available = []
        self.orders_waiting = []
        self.orders_to_deliver = []

