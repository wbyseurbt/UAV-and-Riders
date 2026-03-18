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

    # Calculate distance from order start (current rider pos for picked up orders) to all stations
    # (Here env.r_pos is approximately the start pos if they just picked it up, or en route)
    # Also calculate distance from order end to all stations
    st2end = manhattan(
        env.station_pos.unsqueeze(0).unsqueeze(0),
        c_end.unsqueeze(2),
    )  # (B, R, S)
    closest_dest_sid = st2end.argmin(dim=-1)

    # Heuristic: If the closest station to rider is ALSO the closest station to destination,
    # it implies the stations are not helping to bridge the gap.
    # In this case, force direct delivery (disable case4 forcing).
    useless_hop = (best_sid == closest_dest_sid)

    # NEW: Force direct delivery if useless_hop is True.
    # We define a new case 'case_force_direct' that overrides everything else for these orders.
    case_force_direct = remaining & useless_hop

    frand = torch.rand(B, R, device=dev) < env.force_prob
    # Add useless_hop check to condition (although case_force_direct handles it, keeping it clean here too)
    case4 = remaining & frand & (dist_end > 2 * min_st) & ~useless_hop
    pos4 = env.station_pos[best_sid]
    case4_opt = case4 & ((acts - 1) == best_sid)

    # remaining2 excludes case4 AND case_force_direct
    remaining2 = remaining & ~case4 & ~case_force_direct

    case5a = remaining2 & (acts == 0)
    case5b = remaining2 & (acts > 0)
    sid5 = (acts - 1).clamp(0, S - 1)

    # Re-use st2end computed above
    closest_dest = closest_dest_sid

    # Existing forbid logic: forbid sending to station closest to destination (better to go direct)
    forbid_dest = case5b & (sid5 == closest_dest)
    
    # NEW: Forbid sending to any station that is NOT the closest to the rider (best_sid)
    # i.e., rider can only choose: Direct (acts=0) OR Closest Station (acts=best_sid+1)
    forbid_far = case5b & (sid5 != best_sid)
    
    forbid = forbid_dest | forbid_far
    
    case5b_ok = case5b & ~forbid
    # If forbidden, fallback to closest station (pos4) instead of direct (case5a)
    # UNLESS it was forbidden because it was the destination station (forbid_dest), 
    # in which case we still want direct.
    
    # Logic:
    # 1. forbid_dest -> Force Direct (case5a)
    # 2. forbid_far (but not forbid_dest) -> Force Closest Station (case4/pos4 logic)
    
    # case5a is for Direct Delivery
    # Add forbid_dest to case5a
    case5a = case5a | forbid_dest
    
    # Add forbid_far (that isn't forbid_dest) to a new fallback case that uses pos4
    fallback_to_station = forbid_far & ~forbid_dest
    
    # Note: case4 uses pos4 (closest station). We can merge this fallback into case4 logic 
    # or just use pos4 in the final torch.where.
    # Let's add it to the final construction.
    
    pos5 = env.station_pos[sid5]

    case5b_opt = case5b_ok & (sid5 == best_sid)

    any_case = case1 | case2 | case3 | case4 | case_force_direct | case5a | case5b_ok | fallback_to_station
    new_tgt = torch.zeros(B, R, 2, device=dev)
    new_tgt = torch.where(case1.unsqueeze(-1), pos1, new_tgt)
    new_tgt = torch.where(case2.unsqueeze(-1), c_start, new_tgt)
    new_tgt = torch.where(case3.unsqueeze(-1), c_end, new_tgt)
    new_tgt = torch.where(case4.unsqueeze(-1), pos4, new_tgt)
    new_tgt = torch.where(case_force_direct.unsqueeze(-1), c_end, new_tgt)  # FORCE DIRECT
    new_tgt = torch.where(case5a.unsqueeze(-1), c_end, new_tgt)
    new_tgt = torch.where(case5b_ok.unsqueeze(-1), pos5, new_tgt)
    new_tgt = torch.where(fallback_to_station.unsqueeze(-1), pos4, new_tgt) # FALLBACK TO CLOSEST STATION

    env.r_target = torch.where(any_case.unsqueeze(-1), new_tgt, env.r_target)
    env.r_has_tgt = env.r_has_tgt | any_case

    env.handoff_opt += (case4_opt | case5b_opt).long().sum(dim=-1)
