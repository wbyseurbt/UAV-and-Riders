"""UAV movement, arrival handling and charging for torch vectorized environment (sync-free)."""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import AT_DROP_POINT, UAV_CHARGING, UAV_FLYING, UAV_STOP


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

    env.u_station = torch.where(arrived, tsid, env.u_station)
    env.u_state = torch.where(arrived, torch.tensor(UAV_CHARGING, device=dev, dtype=env.u_state.dtype), env.u_state)
    env.u_tgt = torch.where(arrived, -1, env.u_tgt)
    env.u_bat = torch.where(arrived, (env.u_bat + 0.1).clamp(max=1.0), env.u_bat)

    for k in range(env.uav_cap):
        ok = env.u_orders[:, :, k]  # (B, U)
        valid_ok = (ok >= 0) & arrived
        ok_safe = torch.where(valid_ok, ok, NULL)

        cur_os = env.o_status.gather(1, ok_safe)
        env.o_status.scatter_(1, ok_safe,
            torch.where(valid_ok, torch.tensor(AT_DROP_POINT, device=dev, dtype=cur_os.dtype), cur_os))

        cur_s2 = env.o_s2.gather(1, ok_safe)
        env.o_s2.scatter_(1, ok_safe, torch.where(valid_ok, tsid, cur_s2))

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
