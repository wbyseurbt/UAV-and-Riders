"""GPU-batched vectorized environment for UAV-Rider delivery using PyTorch tensors."""
from __future__ import annotations

import time as _time

import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from uavriders.configs.env_config import EnvConfig, default_config
from uavriders.envs.torch.constants import UAV_STOP
from uavriders.envs.torch.orders import generate_orders
from uavriders.rl.station_actions import process_station_actions
from uavriders.rl.rider_actions import process_rider_actions
from uavriders.envs.torch.rider_movement import move_riders
from uavriders.envs.torch.uav_movement import move_uavs, charge_uavs
from uavriders.envs.torch.last_mile import handle_rider_last_mile
from uavriders.rl.obs import build_obs
from uavriders.rl.rewards import compute_rewards


class TorchVecEnv(VecEnv):
    """Vectorized environment running *num_envs* parallel envs as batched GPU tensors."""

    def __init__(
        self,
        num_envs: int,
        max_steps: int = 200,
        seed: int = 0,
        device: str = "cuda",
        config: EnvConfig | dict | None = None,
        max_orders: int = 500,
        compile: bool = True,
    ):
        if config is None:
            self.cfg = default_config()
        elif isinstance(config, EnvConfig):
            self.cfg = config
        else:
            self.cfg = EnvConfig.from_dict(config)

        self.device = torch.device(device)
        self.max_steps_val = int(max_steps)
        self.max_orders = int(max_orders)
        self._seed_val = seed

        S = self.cfg.n_stations
        R = self.cfg.n_riders

        self._rider_obs_dim = 6 + S * 3
        self._station_obs_dim = 6 + S * 2
        self._obs_len = R * self._rider_obs_dim + S * self._station_obs_dim

        subaction_n = 1 + S
        action_len = R + S

        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_len,), dtype=np.float32
        )
        act_space = spaces.MultiDiscrete([subaction_n] * action_len)
        super().__init__(num_envs, obs_space, act_space)

        self.S = S
        self.R = R
        self.U = self.cfg.n_uavs
        self.grid = self.cfg.world_grid_size
        self.rider_speed = float(self.cfg.rider_speed)
        self.uav_speed = float(self.cfg.uav_speed)
        self.uav_cap = int(self.cfg.uav_capacity_limit)
        self.force_prob = float(self.cfg.force_station_prob)
        self.n_shops = self.cfg.n_shops
        B = num_envs
        dev = self.device
        M = self.max_orders

        self.station_pos = torch.tensor(
            [list(p) for p in self.cfg.station_locs], dtype=torch.float32, device=dev
        )
        self.shop_pos = torch.tensor(
            [list(p) for p in self.cfg.shop_locs], dtype=torch.float32, device=dev
        )

        self.time_t = torch.zeros(B, dtype=torch.long, device=dev)

        self.r_pos = torch.zeros(B, R, 2, dtype=torch.float32, device=dev)
        self.r_free = torch.ones(B, R, dtype=torch.bool, device=dev)
        self.r_carrying = torch.full((B, R), -1, dtype=torch.long, device=dev)
        self.r_pending = torch.full((B, R), -1, dtype=torch.long, device=dev)
        self.r_target = torch.zeros(B, R, 2, dtype=torch.float32, device=dev)
        self.r_has_tgt = torch.zeros(B, R, dtype=torch.bool, device=dev)
        self.r_move_buf = torch.zeros(B, R, dtype=torch.float32, device=dev)

        self.u_pos = torch.zeros(B, self.U, 2, dtype=torch.float32, device=dev)
        self.u_bat = torch.ones(B, self.U, dtype=torch.float32, device=dev)
        self.u_station = torch.zeros(B, self.U, dtype=torch.long, device=dev)
        self.u_tgt = torch.full((B, self.U), -1, dtype=torch.long, device=dev)
        self.u_state = torch.zeros(B, self.U, dtype=torch.long, device=dev)
        self.u_orders = torch.full(
            (B, self.U, self.uav_cap), -1, dtype=torch.long, device=dev
        )
        self.u_ord_cnt = torch.zeros(B, self.U, dtype=torch.long, device=dev)

        # Order tensors: M+1 slots (slot M is a "null" padding slot for sync-free scatter)
        MP = M + 1
        self._null_oid = M
        self.o_start = torch.zeros(B, MP, 2, dtype=torch.float32, device=dev)
        self.o_end = torch.zeros(B, MP, 2, dtype=torch.float32, device=dev)
        self.o_status = torch.zeros(B, MP, dtype=torch.long, device=dev)
        self.o_tcreated = torch.zeros(B, MP, dtype=torch.long, device=dev)
        self.o_twait = torch.zeros(B, MP, dtype=torch.long, device=dev)
        self.o_active = torch.zeros(B, MP, dtype=torch.bool, device=dev)
        self.o_r1 = torch.full((B, MP), -1, dtype=torch.long, device=dev)
        self.o_uav = torch.full((B, MP), -1, dtype=torch.long, device=dev)
        self.o_s1 = torch.full((B, MP), -1, dtype=torch.long, device=dev)
        self.o_s2 = torch.full((B, MP), -1, dtype=torch.long, device=dev)
        self.o_r2 = torch.full((B, MP), -1, dtype=torch.long, device=dev)
        self.o_timedout = torch.zeros(B, MP, dtype=torch.bool, device=dev)
        self.o_count = torch.zeros(B, dtype=torch.long, device=dev)

        self.delivered = torch.zeros(B, dtype=torch.long, device=dev)
        self.uav_launched = torch.zeros(B, dtype=torch.long, device=dev)
        self.uav_balance = torch.zeros(B, dtype=torch.float32, device=dev)
        self.handoff = torch.zeros(B, dtype=torch.long, device=dev)
        self.handoff_opt = torch.zeros(B, dtype=torch.long, device=dev)

        self._ep_rew = torch.zeros(B, dtype=torch.float32, device=dev)
        self._ep_len = torch.zeros(B, dtype=torch.long, device=dev)
        self._ep_start = np.full(B, _time.time())

        self._last_reward_comps: dict[str, torch.Tensor] | None = None

        self._big = torch.tensor(1e9, device=dev)
        self._neg1_f = torch.tensor(-1.0, device=dev)
        self._zero_f = torch.tensor(0.0, device=dev)
        self._penalty = torch.tensor(-200.0, device=dev)
        self._uav_stop_t = torch.tensor(UAV_STOP, dtype=torch.long, device=dev)
        self._st_exclude = torch.eye(S, device=dev) * 1e9

        torch.manual_seed(seed)

        self._actions: np.ndarray | None = None
        self._reset_all()

        # Optional torch.compile (kernel fusion). Safe fallback to eager.
        self._compile = bool(compile)
        if self._compile:
            try:
                self._step_impl = torch.compile(self._step_impl, fullgraph=False)
            except Exception as e:
                print(f"[TorchVecEnv] torch.compile failed, using eager mode: {e}")
                self._compile = False

    def _reset_all(self):
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_envs(mask)

    @torch.no_grad()
    def _reset_envs(self, mask: torch.Tensor):
        idx = torch.where(mask)[0]
        if len(idx) == 0:
            return
        n = len(idx)
        dev = self.device

        self.time_t[idx] = 0

        self.r_pos[idx] = torch.randint(
            0, self.grid + 1, (n, self.R, 2), device=dev, dtype=torch.float32
        )
        self.r_free[idx] = True
        self.r_carrying[idx] = -1
        self.r_pending[idx] = -1
        self.r_target[idx] = 0
        self.r_has_tgt[idx] = False
        self.r_move_buf[idx] = 0

        uav_init_sid = torch.arange(self.U, device=dev) % self.S
        self.u_pos[idx] = self.station_pos[uav_init_sid].unsqueeze(0).expand(n, -1, -1)
        self.u_bat[idx] = 1.0
        self.u_station[idx] = uav_init_sid.unsqueeze(0).expand(n, -1)
        self.u_tgt[idx] = -1
        self.u_state[idx] = UAV_STOP
        self.u_orders[idx] = -1
        self.u_ord_cnt[idx] = 0

        self.o_start[idx] = 0
        self.o_end[idx] = 0
        self.o_status[idx] = 0
        self.o_tcreated[idx] = 0
        self.o_twait[idx] = 0
        self.o_active[idx] = False
        self.o_r1[idx] = -1
        self.o_uav[idx] = -1
        self.o_s1[idx] = -1
        self.o_s2[idx] = -1
        self.o_r2[idx] = -1
        self.o_timedout[idx] = False
        self.o_count[idx] = 0

        self.delivered[idx] = 0
        self.uav_launched[idx] = 0
        self.uav_balance[idx] = 0
        self.handoff[idx] = 0
        self.handoff_opt[idx] = 0

        self._ep_rew[idx] = 0
        self._ep_len[idx] = 0
        now = _time.time()
        for i in idx.cpu().tolist():
            self._ep_start[i] = now

    def reset(self) -> np.ndarray:
        self._reset_all()
        return self._build_obs().cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        actions = torch.from_numpy(self._actions).to(self.device, dtype=torch.long)
        obs, rewards, terminated, truncated, dones = self._step_impl(actions)

        obs_np = obs.cpu().numpy()
        rew_np = rewards.cpu().numpy()
        done_np = dones.cpu().numpy()
        term_np = terminated.cpu().numpy()
        trunc_np = truncated.cpu().numpy()

        B = self.num_envs
        infos: list[dict] = [{} for _ in range(B)]

        if self._last_reward_comps is not None:
            comp_cpu = {k: v.cpu().numpy() for k, v in self._last_reward_comps.items()}
            for i in range(B):
                if trunc_np[i]:
                    infos[i]["reward_components"] = {
                        "overflow_truncate_penalty": -200.0
                    }
                else:
                    infos[i]["reward_components"] = {
                        k: float(v[i]) for k, v in comp_cpu.items()
                    }

        self._ep_rew += rewards
        self._ep_len += 1

        if dones.any():
            terminal_obs = obs_np.copy()
            done_idx = torch.where(dones)[0]

            for i in done_idx.cpu().tolist():
                infos[i]["terminal_observation"] = terminal_obs[i]
                infos[i]["episode"] = {
                    "r": float(self._ep_rew[i].item()),
                    "l": int(self._ep_len[i].item()),
                    "t": _time.time() - self._ep_start[i],
                }
                infos[i]["_episode"] = infos[i]["episode"]
                infos[i]["TimeLimit.truncated"] = bool(trunc_np[i] and not term_np[i])

            self._reset_envs(dones)
            reset_obs = self._build_obs().cpu().numpy()
            obs_np[done_np] = reset_obs[done_np]

            self._ep_rew[done_idx] = 0
            self._ep_len[done_idx] = 0
            now = _time.time()
            for i in done_idx.cpu().tolist():
                self._ep_start[i] = now

        return obs_np, rew_np, done_np, infos

    @torch.no_grad()
    def step_tensor(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, dones = self._step_impl(actions)
        trunc_no_term = truncated & ~terminated

        self._ep_rew += rewards
        self._ep_len += 1

        terminal_obs = None
        ep_info = None

        if dones.any():
            terminal_obs = obs.clone()
            done_idx = torch.where(dones)[0]
            ep_info = (
                done_idx.cpu().numpy(),
                self._ep_rew[done_idx].cpu().numpy(),
                self._ep_len[done_idx].cpu().numpy(),
            )
            self._ep_rew[done_idx] = 0
            self._ep_len[done_idx] = 0
            now = _time.time()
            for i in done_idx.cpu().tolist():
                self._ep_start[i] = now

            self._reset_envs(dones)
            reset_obs = self._build_obs()
            obs = torch.where(dones.unsqueeze(-1), reset_obs, obs)

        return obs, rewards, dones, terminal_obs, trunc_no_term, ep_info

    def _build_obs(self) -> torch.Tensor:
        return build_obs(self)

    @torch.no_grad()
    def _step_impl(self, actions: torch.Tensor):
        B = self.num_envs
        rider_act = actions[:, : self.R]
        station_act = actions[:, self.R :]

        self.time_t += 1

        self.delivered.zero_()
        self.uav_launched.zero_()
        self.uav_balance.zero_()
        self.handoff.zero_()
        self.handoff_opt.zero_()

        self.o_twait.add_(self.o_active.long())

        generate_orders(self)
        process_station_actions(self, station_act)
        process_rider_actions(self, rider_act)
        move_riders(self)
        move_uavs(self)
        charge_uavs(self)
        handle_rider_last_mile(self)

        obs = self._build_obs()
        rewards, comps = compute_rewards(self)
        self._last_reward_comps = comps

        terminated = self.time_t >= self.max_steps_val
        n_active = self.o_active.sum(dim=-1)
        truncated = n_active > 100

        rewards = torch.where(truncated, self._penalty, rewards)
        dones = terminated | truncated
        return obs, rewards, terminated, truncated, dones

    def get_render_state(self, env_index: int = 0):
        """Export state for env_index as a view for MplRenderer (grid_size, stations, riders, uavs, data)."""
        from uavriders.envs.torch.render_view import RenderStateView

        return RenderStateView(self, env_index)

    def close(self):
        pass

    def get_attr(self, attr_name: str, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self, attr_name, None) for _ in indices]

    def set_attr(self, attr_name: str, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return [None for _ in (indices or range(self.num_envs))]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * (
            len(indices) if indices is not None else self.num_envs
        )

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return [seed] * self.num_envs
