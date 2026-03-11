"""Rider action processing for torch vectorized environment."""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import PICKED_BY_R2, UNASSIGNED, manhattan


def process_rider_actions(env, acts: torch.Tensor) -> None:
    """Process rider actions (fully vectorised over B × R)."""
    B, R, S = env.num_envs, env.R, env.S
    dev = env.device

    can_act = ~env.r_has_tgt & (env.r_pending < 0)
    has_order = env.r_carrying >= 0

    soid = env.r_carrying.clamp(min=0)
    c_status = torch.where(
        has_order,
        env.o_status.gather(1, soid),
        torch.full_like(soid, -1),
    )
    c_start = env.o_start.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))
    c_end = env.o_end.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))

    case1 = can_act & ~has_order & (acts > 0)
    sid1 = (acts - 1).clamp(0, S - 1)
    pos1 = env.station_pos[sid1]

    case2 = can_act & has_order & (c_status == UNASSIGNED)
    case3 = can_act & has_order & (c_status == PICKED_BY_R2)

    remaining = (
        can_act
        & has_order
        & (c_status != UNASSIGNED)
        & (c_status != PICKED_BY_R2)
        & (c_status >= 0)
    )

    r2st = manhattan(
        env.r_pos.unsqueeze(2),
        env.station_pos.unsqueeze(0).unsqueeze(0),
    )  # (B, R, S)
    min_st, best_sid = r2st.min(dim=-1)
    dist_end = manhattan(env.r_pos, c_end)

    frand = torch.rand(B, R, device=dev) < env.force_prob
    case4 = remaining & frand & (dist_end > 2 * min_st)
    pos4 = env.station_pos[best_sid]
    case4_opt = case4 & ((acts - 1) == best_sid)

    remaining2 = remaining & ~case4

    case5a = remaining2 & (acts == 0)
    case5b = remaining2 & (acts > 0)
    sid5 = (acts - 1).clamp(0, S - 1)

    st2end = manhattan(
        env.station_pos.unsqueeze(0).unsqueeze(0),
        c_end.unsqueeze(2),
    )  # (B, R, S)
    closest_dest = st2end.argmin(dim=-1)

    forbid = case5b & (sid5 == closest_dest)
    case5b_ok = case5b & ~forbid
    case5a = case5a | forbid
    pos5 = env.station_pos[sid5]

    case5b_opt = case5b_ok & (sid5 == best_sid)

    any_case = case1 | case2 | case3 | case4 | case5a | case5b_ok
    new_tgt = torch.zeros(B, R, 2, device=dev)
    new_tgt = torch.where(case1.unsqueeze(-1), pos1, new_tgt)
    new_tgt = torch.where(case2.unsqueeze(-1), c_start, new_tgt)
    new_tgt = torch.where(case3.unsqueeze(-1), c_end, new_tgt)
    new_tgt = torch.where(case4.unsqueeze(-1), pos4, new_tgt)
    new_tgt = torch.where(case5a.unsqueeze(-1), c_end, new_tgt)
    new_tgt = torch.where(case5b_ok.unsqueeze(-1), pos5, new_tgt)

    env.r_target = torch.where(any_case.unsqueeze(-1), new_tgt, env.r_target)
    env.r_has_tgt = env.r_has_tgt | any_case

    env.handoff_opt += (case4_opt | case5b_opt).long().sum(dim=-1)
