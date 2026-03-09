from __future__ import annotations

import numpy as np

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
    
    # --- Timeout Fallback Logic ---
    # Check for orders that have waited too long in orders_waiting
    # Move them to orders_to_deliver so they can be picked up by riders (fallback)
    # This prevents infinite waiting if no UAV is launched
    # OPTIMIZATION: Reduced threshold from 60 to 20 to react faster to "stuck" orders
    TIMEOUT_THRESHOLD = 20
    timed_out_orders = []
    for oid in station.orders_waiting:
        if env.orders[oid].time_wait > TIMEOUT_THRESHOLD:
            timed_out_orders.append(oid)
    
    if timed_out_orders:
        for oid in timed_out_orders:
            station.orders_waiting.remove(oid)
            station.orders_to_deliver.append(oid)
            # Log or penalize if needed, but for now just fallback
    # -------------------------------

    if not station.uav_available:
        return

    candidates = []
    target_st_pos = env.stations[target_idx].pos
    for oid in station.orders_waiting:
        o = env.orders[oid]
        
        # 只要能缩短距离（让货物离终点更近），就允许发射
        current_dist = manhattan(station.pos, o.end)
        target_dist = manhattan(target_st_pos, o.end)
        
        if target_dist < current_dist:
            candidates.append(oid)

    target_st = env.stations[target_idx]
    
    # --- Enhanced Balance Strategy ---
    # Target is starving if it has very few UAVs OR its backlog is growing relative to its UAV count
    target_needs_help = (len(target_st.uav_available) < 1) or \
                        (len(target_st.orders_waiting) > max(1, len(target_st.uav_available)) * 2)
    
    # Source can give if it has plenty of UAVs OR it has no work to do
    source_can_give = (len(station.uav_available) > 3) or (len(station.orders_waiting) == 0)
    
    should_launch = False
    if candidates:
        should_launch = True
    elif target_needs_help and source_can_give:
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
    
    # Calculate expected battery consumption
    # Distance = Euclidean distance between current station and target station
    current_pos = station.pos.astype(float)
    target_pos = target_st.pos.astype(float)
    dist = np.linalg.norm(target_pos - current_pos)
    
    # Cost = (distance / speed) * (0.001 * speed) = distance * 0.001
    # Or more precisely: steps = ceil(dist / speed), cost = steps * (0.001 * speed)
    # Using continuous approximation consistent with move_uavs:
    # move_uavs reduces 0.001 * speed per step. Total steps approx dist/speed.
    # Total cost approx (dist/speed) * (0.001 * speed) = dist * 0.001
    expected_cost = dist * 0.001
    
    # Ensure battery after flight >= 0.1
    if uav.battery - expected_cost < 0.1:
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

