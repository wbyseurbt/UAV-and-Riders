"""Reward computation for torch vectorized environment."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from uavriders.envs.torch.constants import AT_STATION, UAV_FLYING


def compute_rewards(env) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-env rewards and component dict."""
    B, S, dev = env.num_envs, env.S, env.device

    act_cnt = env.o_active.sum(dim=-1).float()
    active_order_term = -0.0001 * act_cnt
    delivered_term = 8.8 * env.delivered.float()

    tw = (env.o_twait.float() * env.o_active.float()).sum(dim=-1)
    total_wait_term = -0.0001 * tw

    ot = ((env.o_twait > 60) & env.o_active).sum(dim=-1).float()
    overtime_term = -0.2 * ot

    uav_launch_term = (-0.0001 + 0.5) * env.uav_launched.float()

    uav_avail = env.u_state != UAV_FLYING
    us_safe = env.u_station.clamp(0, S - 1)
    u_oh = F.one_hot(us_safe, S).float() * uav_avail.unsqueeze(-1).float()
    ucnt = u_oh.sum(dim=1)  # (B, S)
    excess = (ucnt - env.cfg.station_max_uavs).clamp(min=0).sum(dim=-1)
    overflow_term = -0.1 * excess

    uav_balance_term = 0.5 * env.uav_balance
    handoff_term = 0.05 * env.handoff.float() * env.force_prob
    handoff_opt_term = 0.01 * env.handoff_opt.float() * env.force_prob

    reward = (
        active_order_term
        + delivered_term
        + total_wait_term
        + overtime_term
        + uav_launch_term
        + overflow_term
        + uav_balance_term
        + handoff_term
        + handoff_opt_term
    )

    comps = {
        "active_order_term": active_order_term,
        "delivered_term": delivered_term,
        "total_wait_term": total_wait_term,
        "overtime_term": overtime_term,
        "uav_launch_term": uav_launch_term,
        "overflow_term": overflow_term,
        "uav_balance_term": uav_balance_term,
        "handoff_term": handoff_term,
        "handoff_optimal_term": handoff_opt_term,
    }
    return reward, comps
