"""Rider last-mile assignment — vectorised across B envs AND S stations (sync-free)."""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import AT_DROP_POINT, AT_STATION, manhattan


def handle_rider_last_mile(env) -> None:
    """Assign riders to pick up orders at each station.

    Processes all S stations in a single batch instead of a Python loop.
    """
    B, S, R = env.num_envs, env.S, env.R
    dev = env.device
    NULL = env._null_oid

    sid_row = torch.arange(S, device=dev).unsqueeze(0)  # (1, S)

    # Orders eligible for last-mile pickup at each station: (B, S, Mp)
    o_status_e = env.o_status.unsqueeze(1)
    o_s1_e = env.o_s1.unsqueeze(1)
    o_s2_e = env.o_s2.unsqueeze(1)
    o_active_e = env.o_active.unsqueeze(1)
    o_r2_e = env.o_r2.unsqueeze(1)
    o_timed_e = env.o_timedout.unsqueeze(1)

    sid_col = sid_row.unsqueeze(-1)  # (1, S, 1)

    uav_del = (
        (o_status_e == AT_DROP_POINT)
        & (o_s2_e == sid_col)
        & o_active_e
        & (o_r2_e < 0)
    )
    to_del = (
        (o_status_e == AT_STATION)
        & (o_s1_e == sid_col)
        & o_active_e
        & o_timed_e
        & (o_r2_e < 0)
    )
    orders_pool = uav_del | to_del  # (B, S, Mp)
    has_ord = orders_pool.any(dim=-1)  # (B, S)

    truly_free = (env.r_carrying < 0) & (env.r_pending < 0)  # (B, R)
    has_free = truly_free.any(dim=-1)  # (B,)

    # Station positions expanded for per-station distance: (1, S, 1, 2)
    st_pos_e = env.station_pos.unsqueeze(0).unsqueeze(2)
    d2st = manhattan(env.r_pos.unsqueeze(1), st_pos_e)  # (B, S, R)
    d2st = torch.where(truly_free.unsqueeze(1), d2st, env._big)
    crid = d2st.argmin(dim=-1)  # (B, S) closest free rider per station

    wscore = torch.where(orders_pool, env.o_twait.unsqueeze(1).float(), env._neg1_f)
    boid = wscore.argmax(dim=-1)  # (B, S)

    proc = has_ord & has_free.unsqueeze(1)  # (B, S)

    # ---- scatter rider tensors (B, R) from (B, S) indices ----
    proc_1d = proc.unsqueeze(-1)  # used for 3-arg where

    cur_pend = env.r_pending.gather(1, crid)  # (B, S)
    env.r_pending.scatter_(1, crid, torch.where(proc, boid, cur_pend))

    cur_free = env.r_free.gather(1, crid)
    env.r_free.scatter_(1, crid, torch.where(proc, False, cur_free))

    crid_2d = crid.unsqueeze(-1).expand(-1, -1, 2)
    st_exp = env.station_pos.unsqueeze(0).expand(B, S, 2)
    cur_tgt = env.r_target.gather(1, crid_2d)
    env.r_target.scatter_(1, crid_2d,
        torch.where(proc.unsqueeze(-1).expand_as(cur_tgt), st_exp, cur_tgt))

    cur_htgt = env.r_has_tgt.gather(1, crid)
    env.r_has_tgt.scatter_(1, crid, torch.where(proc, True, cur_htgt))

    # ---- scatter order tensor (B, Mp) from (B, S) indices ----
    boid_safe = torch.where(proc, boid, NULL)
    cur_r2 = env.o_r2.gather(1, boid_safe)
    env.o_r2.scatter_(1, boid_safe, torch.where(proc, crid.long(), cur_r2))
