from __future__ import annotations

import numpy as np

from uavriders.sim.entities import ORDER_STATUS, Order
from uavriders.sim.utils import manhattan


def generate_orders(env) -> None:
    if env._rng.random() < 0.2:
        free_riders = [r for r in env.riders if r.carrying_order is None]
        if not free_riders:
            return

        rider = env._rng.choice(free_riders)

        start_pos = env._rng.choice(env.shop_locs).copy()
        end_pos = np.array([env._rng.randint(0, env.grid_size), env._rng.randint(0, env.grid_size)], dtype=int)

        oid = len(env.orders)
        o = Order(oid, start_pos, end_pos, env.time)
        o.status = ORDER_STATUS["PICKED_BY_R1"]
        o.rider1_id = rider.rid

        env.orders.append(o)
        env.active_orders.append(o)

        rider.pos = start_pos.copy()
        rider.free = False
        rider.carrying_order = o
        rider.target_pos = None


def add_manual_order(env, start_pos, end_pos) -> None:
    free_riders = [r for r in env.riders if r.carrying_order is None]

    if free_riders:
        rider = min(free_riders, key=lambda r: manhattan(r.pos, start_pos))
    else:
        print("无法手动下单：没有空闲骑手！")
        return

    oid = len(env.orders)
    o = Order(oid, start_pos, end_pos, env.time)
    o.status = ORDER_STATUS["PICKED_BY_R1"]
    o.rider1_id = rider.rid

    env.orders.append(o)
    env.active_orders.append(o)

    rider.pos = np.array(start_pos, dtype=int)
    rider.free = False
    rider.carrying_order = o
    rider.target_pos = None

    print(f"手动订单已生成: Order {oid} | {start_pos} -> {end_pos} | Rider {rider.rid}")

