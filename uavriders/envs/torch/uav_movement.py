"""UAV movement, arrival handling and charging for torch vectorized environment (sync-free)."""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import AT_DROP_POINT, AT_STATION, UAV_CHARGING, UAV_FLYING, UAV_STOP, manhattan


def move_uavs(env) -> None:
    """Move flying UAVs toward target station (sync-free)."""
    flying = (env.u_state == UAV_FLYING) & (env.u_tgt >= 0)

    tgt_sid = env.u_tgt.clamp(0, env.S - 1)
    tgt_pos = env.station_pos[tgt_sid].float()

    vec = tgt_pos - env.u_pos
    dist = (vec ** 2).sum(dim=-1).sqrt() + 1e-9

    env.u_bat = torch.where(
        flying,
        (env.u_bat - 0.001 * env.uav_speed).clamp(min=0),
        env.u_bat,
    )

    arrived = flying & (dist <= env.uav_speed)
    still = flying & ~arrived

    direction = vec / dist.unsqueeze(-1)
    new_pos = env.u_pos + direction * env.uav_speed

    env.u_pos = torch.where(
        arrived.unsqueeze(-1),
        tgt_pos,
        torch.where(still.unsqueeze(-1), new_pos, env.u_pos),
    )

    _handle_uav_arrivals(env, arrived)


def _handle_uav_arrivals(env, arrived: torch.Tensor) -> None:
    """Handle UAVs that have reached target station (sync-free)."""
    B, U = env.num_envs, env.U
    dev = env.device
    NULL = env._null_oid

    tsid = env.u_tgt.clamp(min=0)
    
    # Target station position (where UAV arrived)
    tgt_pos = env.station_pos[tsid] # (B, U, 2)

    env.u_station = torch.where(arrived, tsid, env.u_station)
    env.u_state = torch.where(arrived, torch.tensor(UAV_CHARGING, device=dev, dtype=env.u_state.dtype), env.u_state)
    env.u_tgt = torch.where(arrived, -1, env.u_tgt)
    env.u_bat = torch.where(arrived, (env.u_bat + 0.1).clamp(max=1.0), env.u_bat)

    for k in range(env.uav_cap):
        ok = env.u_orders[:, :, k]  # (B, U)
        valid_ok = (ok >= 0) & arrived
        ok_safe = torch.where(valid_ok, ok, NULL)

        # Determine next state for each order: AT_DROP_POINT (last mile) or AT_STATION (transfer)
        # 1. Calculate distance from current station (tsid) to order destination
        # We need to gather o_end for each order in the UAV
        # o_end is (B, Mp, 2)
        # ok_safe is (B, U) -> expand to (B, U, 2) for gather? No, gather on dim 1 (Mp)
        
        # We need to reshape ok_safe to gather from o_end
        # o_end: (B, Mp, 2)
        # ok_safe: (B, U)
        # We want order_dest: (B, U, 2)
        
        # To use gather on o_end (B, Mp, 2), we need indices of shape (B, U, 2) where the last dim is just 0,1
        # But standard gather gathers along one dim.
        # Let's use the expanded gather trick again.
        
        o_end_expanded = env.o_end.unsqueeze(1).expand(-1, U, -1, -1) # (B, U, Mp, 2)
        ok_safe_expanded = ok_safe.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2) # (B, U, 1, 2)
        order_dest = o_end_expanded.gather(2, ok_safe_expanded).squeeze(2) # (B, U, 2)
        
        dist_to_dest = manhattan(tgt_pos, order_dest) # (B, U)
        
        # 2. Calculate distance from ALL stations to order destination to find the absolute closest station
        # st_pos: (S, 2) -> (1, 1, S, 2)
        # order_dest: (B, U, 2) -> (B, U, 1, 2)
        st_pos_exp = env.station_pos.unsqueeze(0).unsqueeze(0) # (1, 1, S, 2)
        order_dest_exp = order_dest.unsqueeze(2) # (B, U, 1, 2)
        
        dists_all_st = manhattan(st_pos_exp, order_dest_exp) # (B, U, S)
        min_dist, closest_sid = dists_all_st.min(dim=-1) # (B, U)
        
        # 3. Decision Logic:
        # If current station (tsid) is the closest station (or very close to it), drop for last mile.
        # Else, keep as AT_STATION for transfer.
        
        is_closest = (tsid == closest_sid)
        
        # If arrived at closest station -> AT_DROP_POINT
        # If not -> AT_STATION (transfer)
        next_status = torch.where(is_closest, 
                                  torch.tensor(AT_DROP_POINT, device=dev, dtype=env.o_status.dtype),
                                  torch.tensor(AT_STATION, device=dev, dtype=env.o_status.dtype))

        cur_os = env.o_status.gather(1, ok_safe)
        env.o_status.scatter_(1, ok_safe, torch.where(valid_ok, next_status, cur_os))

        # Update order location:
        # If AT_STATION (transfer), update o_s1 to current station (waiting for next flight)
        # If AT_DROP_POINT (last mile), update o_s2 to current station (waiting for rider)
        
        # We need to update both potentially, but based on next_status
        # Actually, for AT_STATION, it means it's waiting at this station (o_s1 = tsid)
        # For AT_DROP_POINT, it means it's waiting at this station for rider (o_s2 = tsid, o_s1 is irrelevant/old)
        
        cur_s1 = env.o_s1.gather(1, ok_safe)
        cur_s2 = env.o_s2.gather(1, ok_safe)
        
        new_s1 = torch.where(next_status == AT_STATION, tsid, cur_s1)
        # For AT_DROP_POINT, we set o_s2 to tsid. For AT_STATION, we leave o_s2 alone (or set to -1)
        new_s2 = torch.where(next_status == AT_DROP_POINT, tsid, cur_s2)
        
        env.o_s1.scatter_(1, ok_safe, torch.where(valid_ok, new_s1, cur_s1))
        env.o_s2.scatter_(1, ok_safe, torch.where(valid_ok, new_s2, cur_s2))
        
        # Also, if it is a transfer (AT_STATION), we must reset o_twait to avoid immediate timeout!
        # Because o_twait counts time at current station.
        # If it just arrived, wait time is 0.
        cur_tw = env.o_twait.gather(1, ok_safe)
        env.o_twait.scatter_(1, ok_safe, torch.where(valid_ok & (next_status == AT_STATION), 
                                                     torch.zeros_like(cur_tw), cur_tw))

    env.u_orders = torch.where(arrived.unsqueeze(-1), -1, env.u_orders)
    env.u_ord_cnt = torch.where(arrived, 0, env.u_ord_cnt)


def charge_uavs(env) -> None:
    """Charge UAVs at station."""
    charging = env.u_state == UAV_CHARGING
    env.u_bat = torch.where(
        charging, (env.u_bat + 0.1).clamp(max=1.0), env.u_bat
    )
    full = charging & (env.u_bat >= 1.0)
    env.u_state = torch.where(full, env._uav_stop_t, env.u_state)
