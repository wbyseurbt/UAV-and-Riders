from __future__ import annotations

import numpy as np

from uavriders.sim.station_logic import classify_order_target
from uavriders.sim.utils import manhattan


def get_demand_groups(env, station_obj):
    counts = [0.0] * env.n_stations
    wait_sums = [0.0] * env.n_stations

    current_sid = station_obj.sid

    for oid in station_obj.orders_waiting:
        o = env.orders[oid]
        target_sid = classify_order_target(env, o, current_sid)

        if target_sid != -1:
            counts[target_sid] += 1.0
            wait_sums[target_sid] += float(o.time_wait)

    return counts, wait_sums


def get_obs(env, agent):
    if agent.startswith("rider_"):
        rid = int(agent.split("_")[1])
        r = env.riders[rid]

        x = r.pos[0] / max(1, env.grid_size)
        y = r.pos[1] / max(1, env.grid_size)
        time_norm = min(1.0, env.time / max(1, env.max_steps))

        has_order = 1.0 if r.carrying_order is not None else 0.0

        if r.carrying_order is not None:
            o = r.carrying_order
            wait_norm = min(1.0, o.time_wait / 60.0)
            dist_to_dest = manhattan(r.pos, o.end) / (2 * max(1, env.grid_size))
        else:
            wait_norm = 0.0
            dist_to_dest = 0.0

        base_obs = [x, y, has_order, wait_norm, dist_to_dest, time_norm]

        station_features = []

        MAX_DIST = 2 * max(1, env.grid_size)
        MAX_BUFFER = max(1, env.cfg.station_max_order_buffer)
        MAX_UAVS = max(1, env.cfg.station_max_uavs)

        for st in env.stations:
            d = manhattan(r.pos, st.pos) / MAX_DIST
            cong = len(st.orders_waiting) / MAX_BUFFER
            uavs = len(st.uav_available) / MAX_UAVS
            station_features.extend([d, min(1.0, cong), min(1.0, uavs)])

        return np.array(base_obs + station_features, dtype=np.float32)

    if agent.startswith("station_"):
        sid = int(agent.split("_")[1])
        st = env.stations[sid]

        x = st.pos[0] / max(1, env.grid_size)
        y = st.pos[1] / max(1, env.grid_size)
        time_norm = min(1.0, env.time / max(1, env.max_steps))

        total_waiting_norm = min(1.0, len(st.orders_waiting) / max(1, env.cfg.station_max_order_buffer))
        uav_count_norm = len(st.uav_available) / max(1, env.cfg.station_max_uavs)

        if st.uav_available:
            batteries = [env.uavs[uid].battery for uid in st.uav_available]
            max_battery = max(batteries)
        else:
            max_battery = 0.0

        raw_counts, raw_waits = get_demand_groups(env, st)
        demand_features = []

        MAX_BUFFER = max(1, env.cfg.station_max_order_buffer)
        MAX_TOTAL_WAIT_ESTIMATE = MAX_BUFFER * 60.0

        for k in range(env.n_stations):
            c_norm = min(1.0, raw_counts[k] / MAX_BUFFER)
            w_norm = min(1.0, raw_waits[k] / MAX_TOTAL_WAIT_ESTIMATE)
            demand_features.append(c_norm)
            demand_features.append(w_norm)

        base_obs = [x, y, time_norm, total_waiting_norm, uav_count_norm, max_battery]
        return np.array(base_obs + demand_features, dtype=np.float32)

    return np.zeros((getattr(env, "_station_obs_dim", 12),), dtype=np.float32)

