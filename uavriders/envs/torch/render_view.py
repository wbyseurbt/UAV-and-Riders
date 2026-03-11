"""Render state view: exposes one env's torch state for MplRenderer."""
from __future__ import annotations

import numpy as np
import torch

from uavriders.envs.torch.constants import AT_STATION, UAV_FLYING


class _StationView:
    def __init__(self, pos, uav_available, orders_waiting):
        self.pos = pos
        self.uav_available = uav_available  # list of placeholder (length = count)
        self.orders_waiting = orders_waiting  # list of placeholder (length = count)


class _OrderView:
    __slots__ = ("end",)

    def __init__(self, end):
        self.end = end


class _RiderView:
    def __init__(self, pos, carrying_order=None, target_pos=None):
        self.pos = pos
        self.carrying_order = carrying_order
        self.target_pos = target_pos


class _UAVView:
    def __init__(self, pos, station_id=None, target_station=None):
        self.pos = pos
        self.station_id = station_id  # None if flying
        self.target_station = target_station  # station index when flying


class _DataView:
    """Minimal stats view for render report."""
    def __init__(self):
        self.stats_total_delivered = 0
        self.stats_delivered_by_uav = 0
        self.stats_delivered_by_rider_only = 0
        self.stats_total_delivery_time = 0
        self.stats_uav_delivery_time_sum = 0
        self.stats_rider_delivery_time_sum = 0


class RenderStateView:
    """View of a single env's state for MplRenderer. Reads from TorchVecEnv tensors at env_index."""

    def __init__(self, env, env_index: int = 0):
        self._env = env
        self._i = env_index
        self._data = _DataView()
        self._sync()

    def _sync(self):
        env = self._env
        i = self._i
        dev = env.device

        grid = int(env.grid)
        S, R, U = env.S, env.R, env.U

        # grid_size
        self.grid_size = grid

        # time (for title)
        self.time = int(env.time_t[i].item())

        # shop_locs
        self.shop_locs = env.shop_pos.cpu().numpy().tolist()

        # stations: pos, uav_available (count), orders_waiting (count)
        station_pos = env.station_pos.cpu().numpy()
        u_station = env.u_station[i].cpu()
        u_state = env.u_state[i].cpu()
        at_station = (u_state != UAV_FLYING)
        w_mask = (
            (env.o_status[i] == AT_STATION)
            & env.o_active[i]
            & ~env.o_timedout[i]
        )
        self.stations = []
        for sid in range(S):
            pos = (station_pos[sid][0], station_pos[sid][1])
            n_uav = int(((u_station == sid) & at_station).sum().item())
            n_wait = int((env.o_s1[i] == sid) & w_mask).sum().item()
            self.stations.append(_StationView(pos, [None] * n_uav, [None] * n_wait))

        # riders
        r_pos = env.r_pos[i].cpu().numpy()
        r_carrying = env.r_carrying[i].cpu()
        r_has_tgt = env.r_has_tgt[i].cpu()
        r_target = env.r_target[i].cpu().numpy()
        o_end = env.o_end[i].cpu().numpy()
        self.riders = []
        for r in range(R):
            pos = (float(r_pos[r, 0]), float(r_pos[r, 1]))
            co = None
            if r_carrying[r] >= 0:
                oid = r_carrying[r].item()
                co = _OrderView((float(o_end[oid, 0]), float(o_end[oid, 1])))
            tgt = None
            if r_has_tgt[r]:
                tgt = (float(r_target[r, 0]), float(r_target[r, 1]))
            self.riders.append(_RiderView(pos, co, tgt))

        # uavs
        u_pos = env.u_pos[i].cpu().numpy()
        u_st = env.u_station[i].cpu()
        u_tgt = env.u_tgt[i].cpu()
        u_sta = env.u_state[i].cpu()
        self.uavs = []
        for u in range(U):
            pos = (float(u_pos[u, 0]), float(u_pos[u, 1]))
            sid = None
            tgt_sid = None
            if u_sta[u] == UAV_FLYING:
                tgt_sid = int(u_tgt[u].item()) if u_tgt[u] >= 0 else None
            else:
                sid = int(u_st[u].item()) if u_st[u] >= 0 else None
            self.uavs.append(_UAVView(pos, sid, tgt_sid))

        # data (stats) - optional; we don't track per-episode in torch env, leave at 0
        self.data = self._data


def wrap_torch_env_for_render(torch_vec_env, env_index: int = 0):
    """Return an object that can be passed to MplRenderer.render(env) and has .step/.reset like gym env.
    Caller must call .sync() after each step to refresh the view (or we do it in step).
    """
    class Adapter:
        def __init__(self, env, idx):
            self._env = env
            self._idx = idx
            self._view = None

        def _get_view(self):
            if self._view is None:
                self._view = RenderStateView(self._env, self._idx)
            else:
                self._view._sync()
            return self._view

        @property
        def grid_size(self):
            return self._get_view().grid_size

        @property
        def stations(self):
            return self._get_view().stations

        @property
        def riders(self):
            return self._get_view().riders

        @property
        def uavs(self):
            return self._get_view().uavs

        @property
        def shop_locs(self):
            return self._get_view().shop_locs

        @property
        def data(self):
            return self._get_view().data

        @property
        def time(self):
            return self._get_view().time

        def reset(self, seed=None):
            if seed is not None:
                torch.manual_seed(seed)
            obs = self._env.reset()
            self._view = RenderStateView(self._env, self._idx)
            if isinstance(obs, tuple):
                obs = obs[0]
            return (obs[0] if obs.ndim > 1 else obs, {})

        def step(self, action):
            obs, rewards, dones, infos = self._env.step(np.array([action]))
            self._view = RenderStateView(self._env, self._idx)
            return obs[0], float(rewards[0]), bool(dones[0]), infos[0]

        def close(self):
            self._env.close()

    return Adapter(torch_vec_env, env_index)
