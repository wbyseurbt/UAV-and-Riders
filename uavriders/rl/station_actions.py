"""Station action processing — fully vectorised across B envs AND S stations.

Eliminates the per-station Python loop by broadcasting over an (B, S, …) dimension.
All scatter operations are sync-free (gather + 3-arg torch.where + scatter_).
"""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import AT_STATION, IN_UAV, UAV_FLYING, manhattan


def process_station_actions(env, acts: torch.Tensor) -> None:
    B, S, U = env.num_envs, env.S, env.U
    dev = env.device
    NULL = env._null_oid
    TIMEOUT = 20
    K = min(env.uav_cap, env.max_orders)

    tidx = acts - 1                                         # (B, S)
    sid_row = torch.arange(S, device=dev).unsqueeze(0)      # (1, S)
    valid = (tidx >= 0) & (tidx != sid_row) & (tidx < S)   # (B, S)
    tidx_c = tidx.clamp(0, S - 1)

    # ---- shared expansions (cheap views) ----
    o_status_e = env.o_status.unsqueeze(1)                  # (B, 1, Mp)
    o_s1_e     = env.o_s1.unsqueeze(1)
    o_active_e = env.o_active.unsqueeze(1)
    o_twait_e  = env.o_twait.unsqueeze(1).float()
    sid_col    = sid_row.unsqueeze(-1)                      # (1, S, 1)

    # ---- timeout ----
    waiting_all = (
        (o_status_e == AT_STATION) & (o_s1_e == sid_col) & o_active_e
        & ~env.o_timedout.unsqueeze(1)
    )                                                       # (B, S, Mp)
    timed = waiting_all & (env.o_twait.unsqueeze(1) > TIMEOUT) & valid.unsqueeze(-1)
    env.o_timedout |= timed.any(dim=1)

    o_timed_e = env.o_timedout.unsqueeze(1)
    waiting = (
        (o_status_e == AT_STATION) & (o_s1_e == sid_col) & o_active_e & ~o_timed_e
    )

    # ---- UAVs at source station ----
    u_station_e = env.u_station.unsqueeze(1)                # (B, 1, U)
    u_state_e   = env.u_state.unsqueeze(1)
    uav_at = (u_station_e == sid_row.unsqueeze(-1)) & (u_state_e != UAV_FLYING)  # (B,S,U)
    has_uav = uav_at.any(dim=-1)                            # (B, S)

    bat_m = torch.where(uav_at, env.u_bat.unsqueeze(1).expand(-1, S, -1), env._neg1_f)
    best_bat, best_uid = bat_m.max(dim=-1)                  # (B, S)

    # ---- candidate orders (closer to target station) ----
    o_end_e  = env.o_end.unsqueeze(1)                       # (B, 1, Mp, 2)
    cur_pos  = env.station_pos.unsqueeze(0).unsqueeze(2)    # (1, S, 1, 2)
    tgt_pos  = env.station_pos[tidx_c]                      # (B, S, 2)
    tgt_pos_e = tgt_pos.unsqueeze(2)                        # (B, S, 1, 2)

    cur_dist = (cur_pos - o_end_e).abs().sum(dim=-1)        # (B, S, Mp)
    tgt_dist = (tgt_pos_e - o_end_e).abs().sum(dim=-1)      # (B, S, Mp)
    
    # Relaxed condition: just take ANY waiting order if the agent decided to fly there.
    # We will use uav_optimal_dispatch_term to reward smart choices.
    candidates = waiting 
    
    has_cand = candidates.any(dim=-1)                       # (B, S)

    # ---- Check for optimal dispatch (for reward) ----
    # Did the agent dispatch the UAV to the ABSOLUTE closest station to the order's destination?
    
    # 1. Calculate distance from ALL stations to ALL orders
    # station_pos: (S, 2) -> (1, S, 1, 2)
    # o_end: (B, Mp, 2) -> (B, 1, Mp, 2)
    
    st_pos_exp = env.station_pos.unsqueeze(0).unsqueeze(2) # (1, S, 1, 2)
    o_end_exp = env.o_end.unsqueeze(1)                     # (B, 1, Mp, 2)
    
    # Manhattan broadcasts to (B, S, Mp, 2) and sums last dim -> (B, S, Mp)
    dists_to_all = manhattan(st_pos_exp, o_end_exp)        # (B, S, Mp)
    
    # Find which station is closest for each order (min over S dim)
    # best_st_idx shape: (B, Mp)
    min_dist_to_end, best_st_idx = dists_to_all.min(dim=1) 
    
    # 2. Check if chosen target (tidx) matches the best station
    # tidx: (B, S) -> expand to (B, S, Mp)
    Mp = best_st_idx.shape[-1]
    tidx_exp = tidx.unsqueeze(-1).expand(-1, -1, Mp)       # (B, S, Mp)
    
    # best_st_idx: (B, Mp) -> expand to (B, S, Mp)
    best_st_exp = best_st_idx.unsqueeze(1).expand(-1, S, -1) # (B, S, Mp)
    
    is_optimal_dispatch = (tidx_exp == best_st_exp)        # (B, S, Mp)

    # ---- balance / need heuristic ----
    tgt_uav_at = (u_station_e == tidx_c.unsqueeze(-1)) & (u_state_e != UAV_FLYING)
    tgt_uav_cnt = tgt_uav_at.sum(dim=-1).float()

    tgt_wait = (
        (o_status_e == AT_STATION) & (o_s1_e == tidx_c.unsqueeze(-1))
        & o_active_e & ~o_timed_e
    )
    tgt_wait_cnt = tgt_wait.sum(dim=-1).float()

    tgt_needs = (tgt_uav_cnt < 1) | (tgt_wait_cnt > tgt_uav_cnt.clamp(min=1) * 2)
    src_uav_cnt = uav_at.sum(dim=-1).float()
    src_wait_cnt = waiting.sum(dim=-1).float()
    src_gives = (src_uav_cnt > 3) | (src_wait_cnt == 0)

    bal_trigger = ~has_cand & tgt_needs & src_gives
    should_try = valid & has_uav & (has_cand | bal_trigger)

    # ---- battery feasibility ----
    cur_f = env.station_pos.float().unsqueeze(0)            # (1, S, 2)
    flight_dist = ((tgt_pos.float() - cur_f) ** 2).sum(-1).sqrt()
    cost = flight_dist * 0.001
    bat_ok = (best_bat - cost) >= 0.1
    should_launch = should_try & bat_ok                     # (B, S)

    env.uav_balance += (should_launch & bal_trigger).float().sum(dim=1)

    # ---- top-K order selection (one batched topk) ----
    # score = torch.where(
    #     candidates & should_launch.unsqueeze(-1),
    #     o_twait_e.expand(-1, S, -1),
    #     -env._big,
    # )                                                       # (B, S, Mp)
    
    # NEW SCORING: Prioritize orders that are optimally dispatched!
    # If the chosen station (tidx) matches the order's optimal station, give it a HUGE boost.
    # Otherwise, sort by wait time.
    
    # is_optimal_dispatch: (B, S, Mp)
    
    wait_score = o_twait_e.expand(-1, S, -1).float() # (B, S, Mp)
    # Add a huge constant to optimal orders so they always come first
    optimal_boost = is_optimal_dispatch.float() * 1e6 
    
    final_score = wait_score + optimal_boost
    
    score = torch.where(
        candidates & should_launch.unsqueeze(-1),
        final_score,
        -env._big,
    )
    
    topk_vals, topk_ids = score.topk(K, dim=-1)             # (B, S, K)
    topk_ok = topk_vals > (-env._big + 1)

    # ---- clear UAV order slots ----
    uid_k = best_uid.unsqueeze(-1).expand(-1, -1, env.uav_cap)  # (B, S, cap)
    cur_uo = env.u_orders.gather(1, uid_k)
    launch_k = should_launch.unsqueeze(-1).expand_as(cur_uo)
    env.u_orders.scatter_(1, uid_k, torch.where(launch_k, -1, cur_uo))

    # ---- load orders onto UAV (loop K, sync-free) ----
    IN_UAV_T = torch.tensor(IN_UAV, device=dev, dtype=env.o_status.dtype)
    for k in range(K):
        vk = should_launch & topk_ok[:, :, k]              # (B, S)
        ok = topk_ids[:, :, k]                              # (B, S)
        ok_safe = torch.where(vk, ok, NULL)
        
        # Reward calculation: Check if this specific loaded order was dispatched to its optimal station
        # ok_safe is (B, S), we need to gather from is_optimal_dispatch (B, S, Mp)
        # expand ok_safe to (B, S, 1) for gather
        opt_dispatch_k = is_optimal_dispatch.gather(2, ok_safe.unsqueeze(-1)).squeeze(-1) # (B, S)
        env.uav_optimal_dispatch += (vk & opt_dispatch_k).long().sum(dim=1)

        cur_os = env.o_status.gather(1, ok_safe)
        env.o_status.scatter_(1, ok_safe, torch.where(vk, IN_UAV_T.expand_as(cur_os), cur_os))

        cur_ou = env.o_uav.gather(1, ok_safe)
        env.o_uav.scatter_(1, ok_safe, torch.where(vk, best_uid, cur_ou))

        cur_uok = env.u_orders[:, :, k].gather(1, best_uid)
        env.u_orders[:, :, k].scatter_(1, best_uid, torch.where(vk, ok, cur_uok))

    # ---- finalize UAV state ----
    cnt = topk_ok.sum(dim=-1).long()
    cur_cnt = env.u_ord_cnt.gather(1, best_uid)
    env.u_ord_cnt.scatter_(1, best_uid, torch.where(should_launch, cnt, cur_cnt))

    cur_us = env.u_station.gather(1, best_uid)
    env.u_station.scatter_(1, best_uid, torch.where(should_launch, -1, cur_us))

    cur_ut = env.u_tgt.gather(1, best_uid)
    env.u_tgt.scatter_(1, best_uid, torch.where(should_launch, tidx, cur_ut))

    FLY_T = torch.tensor(UAV_FLYING, device=dev, dtype=env.u_state.dtype)
    cur_ustate = env.u_state.gather(1, best_uid)
    env.u_state.scatter_(1, best_uid,
        torch.where(should_launch, FLY_T.expand_as(cur_ustate), cur_ustate))

    env.uav_launched += should_launch.long().sum(dim=1)
