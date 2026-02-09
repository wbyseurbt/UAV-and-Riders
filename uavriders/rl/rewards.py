from __future__ import annotations


def compute_reward_components(env) -> dict[str, float]:
    active_order_step_penalty = 0.01
    delivered_reward = 0.5
    total_wait_penalty_coef = 0.0001
    overtime_threshold = 60
    overtime_penalty = 1.0
    uav_launch_cost = 0.0001
    hub_overflow_penalty_coef = 0.02
    uav_order_balance_reward_coef = 0.003
    handoff_reward_coef = 0.05
    handoff_optimal_reward_coef = 0.01

    active_order_term = -active_order_step_penalty * float(len(env.active_orders))
    delivered_term = delivered_reward * float(env._delivered_this_step)

    total_wait = float(sum(o.time_wait for o in env.active_orders))
    total_wait_term = -total_wait_penalty_coef * total_wait

    overtime = float(sum(1 for o in env.active_orders if o.time_wait > overtime_threshold))
    overtime_term = -overtime_penalty * overtime

    uav_launch_term = -uav_launch_cost * float(env._uav_launch_this_step)

    overflow = 0
    for st in env.stations:
        if len(st.uav_available) > st.max_uavs:
            overflow += len(st.uav_available) - st.max_uavs
    overflow_term = -hub_overflow_penalty_coef * float(overflow) if overflow > 0 else 0.0

    uav_balance_term = uav_order_balance_reward_coef * float(env._uav_order_balance)
    handoff_term = handoff_reward_coef * float(env._handoff_this_step) * float(env.force_station_prob)
    handoff_optimal_term = (
        handoff_optimal_reward_coef
        * float(env._handoff_optimal_this_step)
        * float(env.force_station_prob)
    )

    return {
        "active_order_term": float(active_order_term),
        "delivered_term": float(delivered_term),
        "total_wait_term": float(total_wait_term),
        "overtime_term": float(overtime_term),
        "uav_launch_term": float(uav_launch_term),
        "overflow_term": float(overflow_term),
        "uav_balance_term": float(uav_balance_term),
        "handoff_term": float(handoff_term),
        "handoff_optimal_term": float(handoff_optimal_term),
    }


def compute_shared_reward(env) -> float:
    components = compute_reward_components(env)
    return float(sum(components.values()))

