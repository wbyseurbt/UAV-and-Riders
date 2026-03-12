"""Render state view: exposes one env's torch state for MplRenderer."""
from __future__ import annotations

import numpy as np
import torch

from uavriders.envs.torch.constants import AT_STATION, DELIVERED, UAV_FLYING


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
    def __init__(self, stats_dict=None):
        self.stats_total_delivered = stats_dict.get("delivered", 0) if stats_dict else 0
        self.stats_delivered_by_uav = stats_dict.get("delivered_by_uav", 0) if stats_dict else 0
        self.stats_delivered_by_rider_only = stats_dict.get("delivered_by_rider_only", 0) if stats_dict else 0
        self.stats_total_delivery_time = stats_dict.get("total_delivery_time", 0) if stats_dict else 0
        self.stats_uav_delivery_time_sum = stats_dict.get("uav_delivery_time_sum", 0) if stats_dict else 0
        self.stats_rider_delivery_time_sum = stats_dict.get("rider_delivery_time_sum", 0) if stats_dict else 0


class RenderStateView:
    """View of a single env's state for MplRenderer. Reads from TorchVecEnv tensors at env_index."""

    def __init__(self, env, env_index: int = 0):
        self._env = env
        self._i = env_index
        self._data = None # Will be populated in _sync
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
        self.shop_locs = []
        shop_pos = env.shop_pos.cpu().numpy()
        for k in range(len(shop_pos)):
             self.shop_locs.append((float(shop_pos[k][0]), float(shop_pos[k][1])))

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
            pos = (float(station_pos[sid][0]), float(station_pos[sid][1]))
            
            # 安全获取 n_uav
            # u_station shape [U], at_station shape [U]
            # u_station == sid -> [U] bool
            mask_uav = (u_station == sid) & at_station
            n_uav = int(mask_uav.sum().item())
            
            # 安全获取 n_wait
            # env.o_s1 shape [n_envs, n_orders], o_s1[i] shape [n_orders]
            # w_mask shape [n_orders]
            # mask -> [n_orders] bool
            s1_i = env.o_s1[i].cpu() # move to cpu first for safety
            mask_wait = (s1_i == sid) & w_mask.cpu()
            n_wait = int(mask_wait.sum().item())
            
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
        u_target = env.u_tgt[i].cpu()
        u_sta = env.u_state[i].cpu()
        self.uavs = []
        for u in range(U):
            pos = (float(u_pos[u, 0]), float(u_pos[u, 1]))
            sid = None
            tgt_sid = None
            if u_sta[u] == UAV_FLYING:
                if u_target[u] >= 0:
                    tgt_sid = int(u_target[u].item())
            else:
                if u_st[u] >= 0:
                    sid = int(u_st[u].item())
            self.uavs.append(_UAVView(pos, sid, tgt_sid))

        # data (stats) - populated from env tensors
        # Note: In TorchVecEnv, we track stats cumulatively in tensors.
        # Here we just grab the current values for the specific env index.
        stats_dict = {
            "delivered": int(env.delivered[i].item()),
            "delivered_by_uav": int(env.uav_launched[i].item()),  # Approximate proxy if not tracking separately
            "delivered_by_rider_only": int(env.delivered[i].item()) - int(env.uav_launched[i].item()), # Proxy
            # Note: The TorchVecEnv might not track delivery times per episode yet.
            # We will use placeholders if not available.
            "total_delivery_time": 0, 
            "uav_delivery_time_sum": 0,
            "rider_delivery_time_sum": 0,
        }
        self.data = _DataView(stats_dict)


def wrap_torch_env_for_render(torch_vec_env, env_index: int = 0):
    """Return an object that can be passed to MplRenderer.render(env) and has .step/.reset like gym env.
    Caller must call .sync() after each step to refresh the view (or we do it in step).
    """
    class Adapter:
        def __init__(self, env, idx):
            self._env = env
            self._idx = idx
            self._view = None
            
            # Accumulators for stats (since env tensors are reset per step)
            self._total_delivered = 0
            self._total_uav_launched = 0
            self._total_steps = 0

            # Detailed delivery stats
            self._stats_rider_only_count = 0
            self._stats_uav_assist_count = 0
            self._stats_rider_only_time_sum = 0
            self._stats_uav_assist_time_sum = 0
            self._delivered_oids = set()

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
            
            # Reset accumulators
            self._total_delivered = 0
            self._total_uav_launched = 0
            self._total_steps = 0
            
            self._stats_rider_only_count = 0
            self._stats_uav_assist_count = 0
            self._stats_rider_only_time_sum = 0
            self._stats_uav_assist_time_sum = 0
            self._delivered_oids = set()

            if isinstance(obs, tuple):
                obs = obs[0]
            return (obs[0] if obs.ndim > 1 else obs, {})

        def step(self, action):
            obs, rewards, dones, infos = self._env.step(np.array([action]))
            
            # Update accumulators from current step's events
            idx = self._idx
            # delivered is per-step count in env
            step_delivered = int(self._env.delivered[idx].item())
            step_uav = int(self._env.uav_launched[idx].item())
            
            self._total_delivered += step_delivered
            self._total_uav_launched += step_uav
            self._total_steps += 1

            # Detailed stats tracking
            current_time = self._env.time_t[idx].item()
            n_orders = self._env.o_count[idx].item()
            
            o_status = self._env.o_status[idx].cpu()
            o_uav = self._env.o_uav[idx].cpu()
            o_tcreated = self._env.o_tcreated[idx].cpu()
            
            # Check newly delivered orders
            # Only check if step_delivered > 0 to save time
            if step_delivered > 0:
                # Find orders that are DELIVERED but not yet counted
                # (This loop could be optimized if n_orders is huge, but here it's fine)
                for oid in range(n_orders):
                    if o_status[oid] == DELIVERED and oid not in self._delivered_oids:
                        self._delivered_oids.add(oid)
                        
                        delivery_time = float(current_time - o_tcreated[oid])
                        is_uav = (o_uav[oid] >= 0)
                        
                        if is_uav:
                            self._stats_uav_assist_count += 1
                            self._stats_uav_assist_time_sum += delivery_time
                        else:
                            self._stats_rider_only_count += 1
                            self._stats_rider_only_time_sum += delivery_time

            self._view = RenderStateView(self._env, self._idx)
            # Override the data view with our accumulated stats
            stats_dict = {
                "delivered": self._total_delivered,
                "delivered_by_uav": self._stats_uav_assist_count,
                "delivered_by_rider_only": self._stats_rider_only_count,
                "total_delivery_time": self._stats_rider_only_time_sum + self._stats_uav_assist_time_sum, 
                "uav_delivery_time_sum": self._stats_uav_assist_time_sum,
                "rider_delivery_time_sum": self._stats_rider_only_time_sum,
            }
            self._view.data = _DataView(stats_dict)
            
            # Unpack done -> terminated, truncated
            done = bool(dones[0])
            info = infos[0]
            # Try to get truncated from info, otherwise assume terminated
            truncated = info.get("TimeLimit.truncated", False)
            terminated = done and not truncated
            
            return obs[0], float(rewards[0]), terminated, truncated, info

        def close(self):
            self._env.close()

    return Adapter(torch_vec_env, env_index)
