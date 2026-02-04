from __future__ import annotations

from uavriders.sim.entities import ORDER_STATUS
from uavriders.sim.utils import manhattan


def classify_order_target(env, order_obj, current_sid: int) -> int:
    best_sid = -1
    min_dist = float("inf")

    for st in env.stations:
        if st.sid == current_sid:
            continue

        d = manhattan(st.pos, order_obj.end)
        if d < min_dist:
            min_dist = d
            best_sid = st.sid

    return best_sid


def process_station_action(env, sid: int, action) -> None:
    target_idx = int(action) - 1

    if target_idx < 0:
        return
    if target_idx == sid:
        return
    if target_idx >= env.n_stations:
        return

    station = env.stations[sid]

    if not station.uav_available:
        return

    candidates = []
    for oid in station.orders_waiting:
        o = env.orders[oid]
        best_gateway = classify_order_target(env, o, sid)
        if best_gateway == target_idx:
            candidates.append(oid)

    target_st = env.stations[target_idx]
    target_is_starving = (len(target_st.uav_available) == 0) and (len(target_st.orders_waiting) > 0)
    should_launch = False
    if candidates:
        should_launch = True
    elif target_is_starving:
        should_launch = True
        env._uav_order_balance += 1.0

    if not should_launch:
        return

    uav_battery_max = -1.0
    uav_battery_max_id = -1
    for uav_id in station.uav_available:
        uav = env.uavs[uav_id]
        if uav.battery >= uav_battery_max:
            uav_battery_max = uav.battery
            uav_battery_max_id = uav_id

    uav_id = uav_battery_max_id
    uav = env.uavs[uav_id]
    if uav.battery < 0.1:
        return

    candidates.sort(key=lambda oid: env.orders[oid].time_wait, reverse=True)
    oids_to_load = candidates[: uav.capacity_limit]

    station.uav_available.remove(uav_id)
    station.orders_waiting = [x for x in station.orders_waiting if x not in oids_to_load]

    for oid in oids_to_load:
        o = env.orders[oid]
        o.status = ORDER_STATUS["IN_UAV"]
        o.uav_id = uav_id

    uav.orders = list(oids_to_load)
    uav.station_id = None
    uav.target_station = target_idx
    uav.state = "FLYING"

    env._uav_launch_this_step += 1

