"""Observation building for torch vectorized environment (sync-free)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from uavriders.envs.torch.constants import AT_STATION, UAV_FLYING, manhattan


def build_obs(env) -> torch.Tensor:
    """Build concatenated observation tensor (B, obs_dim), sync-free."""
    B, R, S = env.num_envs, env.R, env.S
    dev = env.device
    grid = max(1, env.grid)

    uav_avail = env.u_state != UAV_FLYING
    us_safe = env.u_station.clamp(0, S - 1)
    u_oh = F.one_hot(us_safe, S).float() * uav_avail.unsqueeze(-1).float()
    uav_cnt = u_oh.sum(dim=1)  # (B, S)

    w_mask = (
        (env.o_status == AT_STATION)
        & env.o_active
        & ~env.o_timedout
    )
    os1_safe = env.o_s1.clamp(0, S - 1)
    o_oh = F.one_hot(os1_safe, S).float() * w_mask.unsqueeze(-1).float()
    cong = o_oh.sum(dim=1)  # (B, S)

    bat_exp = env.u_bat.unsqueeze(-1) * u_oh
    max_bat = bat_exp.max(dim=1).values.clamp(min=0)

    MAX_BUF = float(max(1, env.cfg.station_max_order_buffer))
    MAX_UAVS = float(max(1, env.cfg.station_max_uavs))
    MAX_DIST = 2.0 * grid

    rx = env.r_pos[..., 0] / grid
    ry = env.r_pos[..., 1] / grid
    tn = (env.time_t.float() / max(1, env.max_steps_val)).unsqueeze(1).expand(B, R)
    ho = (env.r_carrying >= 0).float()

    soid = env.r_carrying.clamp(min=0)
    c_wait = env.o_twait.gather(1, soid).float()
    c_end = env.o_end.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))
    w_norm = (c_wait / 60.0).clamp(max=1.0) * ho
    d_dest = manhattan(env.r_pos, c_end).float() / MAX_DIST * ho

    r_base = torch.stack([rx, ry, ho, w_norm, d_dest, tn], dim=-1)

    r2s = manhattan(
        env.r_pos.unsqueeze(2),
        env.station_pos.unsqueeze(0).unsqueeze(0),
    ).float() / MAX_DIST

    cong_n = (cong / MAX_BUF).clamp(max=1.0).unsqueeze(1).expand(B, R, S)
    uav_n = (uav_cnt / MAX_UAVS).clamp(max=1.0).unsqueeze(1).expand(B, R, S)
    s_feat = torch.stack([r2s, cong_n, uav_n], dim=-1).reshape(B, R, S * 3)

    r_obs = torch.cat([r_base, s_feat], dim=-1).reshape(B, R * env._rider_obs_dim)

    sx = env.station_pos[:, 0] / grid
    sy = env.station_pos[:, 1] / grid
    sxy = torch.stack([sx, sy], dim=-1).unsqueeze(0).expand(B, S, 2)
    tn_s = (env.time_t.float() / max(1, env.max_steps_val)).unsqueeze(1).expand(B, S)
    tw_n = (cong / MAX_BUF).clamp(max=1.0)
    uc_n = (uav_cnt / MAX_UAVS).clamp(max=1.0)

    s_base = torch.stack(
        [sxy[..., 0], sxy[..., 1], tn_s, tw_n, uc_n, max_bat], dim=-1
    )

    MAX_WAIT_EST = MAX_BUF * 60.0

    # Vectorised demand across all source stations (no per-station loop)
    d2s_all = manhattan(
        env.o_end.unsqueeze(2),
        env.station_pos.unsqueeze(0).unsqueeze(0),
    )  # (B, Mp, S)

    # wmask per source station: (B, S, Mp)
    wmask_all = w_mask.unsqueeze(1) & (env.o_s1.unsqueeze(1) == torch.arange(S, device=dev).unsqueeze(0).unsqueeze(-1))

    # Exclude self-station from argmin: (B, S, Mp, S)
    d2s_ex = d2s_all.unsqueeze(1) + env._st_exclude.unsqueeze(0).unsqueeze(2)
    tsid = d2s_ex.argmin(dim=-1)  # (B, S, Mp)

    t_oh = F.one_hot(tsid, S).float() * wmask_all.unsqueeze(-1).float()  # (B, S, Mp, S)
    cnt_k = t_oh.sum(dim=2)  # (B, S, S)
    wt_k = (env.o_twait.float().unsqueeze(1).unsqueeze(-1) * t_oh).sum(dim=2)  # (B, S, S)

    demand = torch.zeros(B, S, S * 2, device=dev)
    demand[:, :, 0::2] = (cnt_k / MAX_BUF).clamp(max=1.0)
    demand[:, :, 1::2] = (wt_k / MAX_WAIT_EST).clamp(max=1.0)

    s_obs = torch.cat([s_base, demand], dim=-1).reshape(B, S * env._station_obs_dim)

    return torch.cat([r_obs, s_obs], dim=-1)
