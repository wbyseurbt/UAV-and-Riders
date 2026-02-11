from __future__ import annotations

import numpy as np

from uavriders.sim.entities import ORDER_STATUS
from uavriders.sim.utils import manhattan


def process_rider_action(env, rid: int, action: int) -> None:
    rider = env.riders[rid]
    if rider.target_pos is not None:
        return

    # If the rider is heading to pick up a pending order (last mile), ignore strategy actions
    if rider.pending_order is not None:
        return

    if rider.carrying_order is None:
        if action > 0:
            sid = action - 1
            if 0 <= sid < env.n_stations:
                rider.target_pos = env.stations[sid].pos.copy()
        return

    order = rider.carrying_order
    if order.status == ORDER_STATUS["UNASSIGNED"]:
        rider.target_pos = order.start.copy()
        return
    is_last_mile = order.status == ORDER_STATUS["PICKED_BY_R2"]
    force_station = env._rng.random() < env.force_station_prob

    if is_last_mile:
        rider.target_pos = order.end.copy()
        return

    if force_station:
        best_sid = -1
        min_dist = float("inf")
        for st in env.stations:
            d = manhattan(rider.pos, st.pos)
            if d < min_dist:
                min_dist = d
                best_sid = st.sid

        # Only force station if delivery distance is significantly larger than distance to station
        # e.g., delivery distance > 1.5 * distance to station
        dist_to_end = manhattan(rider.pos, order.end)
        if best_sid != -1 and dist_to_end > 2 * min_dist:
            rider.target_pos = env.stations[best_sid].pos.copy()
            
            # Check if optimal action was chosen (for reward stats)
            target_sid = action - 1
            if target_sid == best_sid:
                env._handoff_optimal_this_step += 1
            return
        # If condition not met, fall through to normal strategy logic

    if action == 0:
        rider.target_pos = order.end.copy()
        return

    target_sid = action - 1
    if 0 <= target_sid < env.n_stations:
        rider.target_pos = env.stations[target_sid].pos.copy()

    best_sid = -1
    min_dist = float("inf")
    for st in env.stations:
        d = manhattan(rider.pos, st.pos)
        if d < min_dist:
            min_dist = d
            best_sid = st.sid
    if best_sid != -1 and target_sid == best_sid:
        env._handoff_optimal_this_step += 1


def move_riders(env) -> None:
    for rider in env.riders:
        if rider.target_pos is None:
            continue

        if np.array_equal(rider.pos, rider.target_pos):
            handle_rider_arrival(env, rider)
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
                handle_rider_arrival(env, rider)
                rider.move_buffer = 0.0


def handle_rider_arrival(env, rider) -> None:
    if rider.carrying_order is None:
        if rider.pending_order is not None:
            for st in env.stations:
                if np.array_equal(rider.pos, st.pos):
                    oid = int(rider.pending_order)
                    o = env.orders[oid]
                    o.status = ORDER_STATUS["PICKED_BY_R2"]
                    o.rider2_id = rider.rid
                    rider.carrying_order = o
                    rider.pending_order = None
                    rider.target_pos = o.end.copy()
                    rider.free = False
                    return
        rider.target_pos = None
        rider.free = True
        return

    o = rider.carrying_order

    if o.status == ORDER_STATUS["UNASSIGNED"] and np.array_equal(rider.pos, o.start):
        o.status = ORDER_STATUS["PICKED_BY_R1"]
        rider.target_pos = None
        rider.free = False
        return

    if np.array_equal(rider.pos, o.end):
        o.status = ORDER_STATUS["DELIVERED"]
        env._delivered_this_step += 1
        if o in env.active_orders:
            env.active_orders.remove(o)

        rider.carrying_order = None
        rider.target_pos = None
        rider.free = True
        return

    if o.status == ORDER_STATUS["PICKED_BY_R1"]:
        for st in env.stations:
            if np.array_equal(rider.pos, st.pos):
                if o.oid not in st.orders_waiting:
                    st.orders_waiting.append(o.oid)

                o.station1_id = st.sid
                o.status = ORDER_STATUS["AT_STATION"]

                rider.carrying_order = None
                rider.target_pos = None
                rider.free = True

                env._handoff_this_step += 1
                return

    if rider.carrying_order is not None and rider.target_pos is None:
        rider.target_pos = rider.carrying_order.end.copy()


def handle_rider_last_mile(env, station) -> None:
    if not station.orders_to_deliver:
        return

    # Filter riders who are truly free (no carrying order AND no pending order)
    free_riders = [r for r in env.riders if r.carrying_order is None and r.pending_order is None]
    if not free_riders:
        return

    rider = min(free_riders, key=lambda r: manhattan(r.pos, station.pos))
    oid = max(station.orders_to_deliver, key=lambda x: env.orders[x].time_wait)
    station.orders_to_deliver.remove(oid)

    o = env.orders[oid]
    rider.pending_order = oid
    rider.free = False
    rider.target_pos = station.pos.copy()
