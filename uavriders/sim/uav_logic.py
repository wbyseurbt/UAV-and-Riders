from __future__ import annotations

import numpy as np

from uavriders.sim.entities import ORDER_STATUS


def move_uavs(env) -> None:
    for uav in env.uavs:
        if uav.station_id is None and uav.target_station is not None:
            target_pos = env.stations[uav.target_station].pos.astype(float)
            vec = target_pos - uav.pos
            dist = float(np.linalg.norm(vec) + 1e-9)

            uav.battery = max(0.0, uav.battery - 0.001 * uav.speed)

            if dist <= uav.speed:
                uav.pos = target_pos
                handle_uav_arrival(env, uav)
            else:
                uav.pos = uav.pos + (vec / dist) * uav.speed


def handle_uav_arrival(env, uav) -> None:
    dest_station = env.stations[uav.target_station]

    dest_station.uav_available.append(uav.uid)
    dest_station.orders_to_deliver.extend(uav.orders)

    for oid in uav.orders:
        o = env.orders[oid]
        o.status = ORDER_STATUS["AT_DROP_POINT"]
        o.station2_id = dest_station.sid

    uav.orders = []
    uav.state = "CHARGING"
    uav.station_id = dest_station.sid
    uav.target_station = None

    uav.battery = min(1.0, uav.battery + 0.1)


def charge_uavs(env) -> None:
    for uav in env.uavs:
        if uav.state == "CHARGING":
            uav.battery = min(1.0, uav.battery + 0.1)
            if uav.battery >= 1.0:
                uav.state = "STOP"

