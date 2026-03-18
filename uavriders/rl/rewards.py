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

    uav_launch_term = -0.01 * env.uav_launched.float()

    # --- UAV Optimal Dispatch Bonus ---
    # Give a large one-time bonus if the agent dispatches a UAV directly to the optimal station
    # (the station closest to the order's destination).
    uav_optimal_dispatch_term = 0.8 * env.uav_optimal_dispatch.float()

   
    
    timeout_count = (env.o_timedout & env.o_active).float().sum(dim=-1)
    timeout_penalty_term = -0.2 * timeout_count # Heavy penalty per step per timed-out order!

    # --- UAV Delivery Bonus (Payload Distance Reduction) ---
    # We reward UAVs for moving orders closer to their destination.
    # This is a dense reward calculated at every step for flying UAVs carrying orders.
    
    # 1. Identify flying UAVs with orders
    flying = (env.u_state == UAV_FLYING) & (env.u_tgt >= 0) # (B, U)
    
    # 2. Calculate progress towards destination
    # Current UAV position
    u_pos = env.u_pos  # (B, U, 2)
    
    # Target Station Position (where UAV is flying to)
    tgt_sid = env.u_tgt.clamp(0, S - 1)
    tgt_pos = env.station_pos[tgt_sid].float() # (B, U, 2)
    
    # Vector from current pos to target
    vec_to_tgt = tgt_pos - u_pos
    dist_to_tgt = (vec_to_tgt ** 2).sum(dim=-1).sqrt() + 1e-9
    
    # Normalized direction vector
    u_dir = vec_to_tgt / dist_to_tgt.unsqueeze(-1) # (B, U, 2)
    
    # Distance moved in this step = speed * direction
    # (Assuming UAV always moves at max speed towards target)
    step_move = u_dir * env.uav_speed # (B, U, 2)
    
    # 3. For each order in each UAV, calculate if this move reduces distance to order's end
    # This requires gathering order destinations.
    # u_orders: (B, U, Cap) -> indices of orders
    # o_end: (B, Mp, 2) -> order destinations
    
    # Expand u_orders for gathering
    u_ord_idx = env.u_orders.long() # (B, U, Cap)
    mask_valid = (u_ord_idx >= 0) & flying.unsqueeze(-1) # Only valid orders in flying UAVs
    
    # Safe index for gather
    safe_idx = u_ord_idx.clamp(min=0)
    
    # Gather order destinations: (B, U, Cap, 2)
    # We need to expand o_end to (B, 1, Mp, 2) then gather
    o_end_expanded = env.o_end.unsqueeze(1).expand(-1, env.U, -1, -1) # (B, U, Mp, 2)
    
    # Gather requires index to have same dims as output except on dim being gathered
    # We want to gather along dim=2 (Mp) using safe_idx (B, U, Cap)
    # So we need to replicate safe_idx on the last dim (coordinates)
    safe_idx_expanded = safe_idx.unsqueeze(-1).expand(-1, -1, -1, 2) # (B, U, Cap, 2)
    
    order_dests = o_end_expanded.gather(2, safe_idx_expanded) # (B, U, Cap, 2)
    
    # Current distance from UAV to Order Dest
    # u_pos is (B, U, 2) -> expand to (B, U, 1, 2)
    curr_dist = (u_pos.unsqueeze(2) - order_dests).norm(dim=-1) # (B, U, Cap)
    
    # Next position (after this step)
    next_pos = u_pos + step_move # (B, U, 2)
    next_dist = (next_pos.unsqueeze(2) - order_dests).norm(dim=-1) # (B, U, Cap)
    
    # Progress = reduction in distance
    progress = (curr_dist - next_dist) # (B, U, Cap)
    
    # Reward: Sum of progress for all valid orders
    # We clip negative progress to 0 (don't punish if slightly off-angle, though usually it's positive)
    # Scale factor 0.1
    uav_transport_reward = (progress * mask_valid.float()).sum(dim=(1, 2)) * 0.01


    uav_avail = env.u_state != UAV_FLYING
    us_safe = env.u_station.clamp(0, S - 1)
    u_oh = F.one_hot(us_safe, S).float() * uav_avail.unsqueeze(-1).float()
    ucnt = u_oh.sum(dim=1)  # (B, S)
    excess = (ucnt - env.cfg.station_max_uavs).clamp(min=0).sum(dim=-1)
    overflow_term = -0.1 * excess

    uav_balance_term = 0.05 * env.uav_balance
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
        + uav_transport_reward
        + uav_optimal_dispatch_term
        + timeout_penalty_term
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
        "uav_transport_reward": uav_transport_reward,
        "uav_optimal_dispatch_term": uav_optimal_dispatch_term,
        "timeout_penalty_term": timeout_penalty_term,
    }
    return reward, comps
