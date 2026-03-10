"""GPU-batched vectorized environment for UAV-Rider delivery using PyTorch tensors.

Replaces SubprocVecEnv + N Python environment processes with a single process
running all N environments as batched tensor operations on GPU (or CPU).
"""
from __future__ import annotations

import time as _time

import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from uavriders.configs.env_config import EnvConfig, default_config

# ---------------------------------------------------------------------------
# Order status constants (must match entities.ORDER_STATUS)
# ---------------------------------------------------------------------------
UNASSIGNED = 0
PICKED_BY_R1 = 1
AT_STATION = 2
IN_UAV = 3
AT_DROP_POINT = 4
PICKED_BY_R2 = 5
DELIVERED = 6

UAV_STOP = 0
UAV_FLYING = 1
UAV_CHARGING = 2


def _manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched manhattan distance along the last dimension (size 2)."""
    return (a - b).abs().sum(dim=-1)


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

        # Shorthand config
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

        # Fixed positions (shared across envs)
        self.station_pos = torch.tensor(
            [list(p) for p in self.cfg.station_locs], dtype=torch.float32, device=dev
        )  # (S, 2)
        self.shop_pos = torch.tensor(
            [list(p) for p in self.cfg.shop_locs], dtype=torch.float32, device=dev
        )  # (n_shops, 2)

        # --- State tensors ---
        self.time_t = torch.zeros(B, dtype=torch.long, device=dev)

        # Riders
        self.r_pos = torch.zeros(B, R, 2, dtype=torch.float32, device=dev)
        self.r_free = torch.ones(B, R, dtype=torch.bool, device=dev)
        self.r_carrying = torch.full((B, R), -1, dtype=torch.long, device=dev)
        self.r_pending = torch.full((B, R), -1, dtype=torch.long, device=dev)
        self.r_target = torch.zeros(B, R, 2, dtype=torch.float32, device=dev)
        self.r_has_tgt = torch.zeros(B, R, dtype=torch.bool, device=dev)
        self.r_move_buf = torch.zeros(B, R, dtype=torch.float32, device=dev)

        # UAVs
        self.u_pos = torch.zeros(B, self.U, 2, dtype=torch.float32, device=dev)
        self.u_bat = torch.ones(B, self.U, dtype=torch.float32, device=dev)
        self.u_station = torch.zeros(B, self.U, dtype=torch.long, device=dev)
        self.u_tgt = torch.full((B, self.U), -1, dtype=torch.long, device=dev)
        self.u_state = torch.zeros(B, self.U, dtype=torch.long, device=dev)
        self.u_orders = torch.full(
            (B, self.U, self.uav_cap), -1, dtype=torch.long, device=dev
        )
        self.u_ord_cnt = torch.zeros(B, self.U, dtype=torch.long, device=dev)

        # Orders (pre-allocated buffer)
        self.o_start = torch.zeros(B, M, 2, dtype=torch.float32, device=dev)
        self.o_end = torch.zeros(B, M, 2, dtype=torch.float32, device=dev)
        self.o_status = torch.zeros(B, M, dtype=torch.long, device=dev)
        self.o_tcreated = torch.zeros(B, M, dtype=torch.long, device=dev)
        self.o_twait = torch.zeros(B, M, dtype=torch.long, device=dev)
        self.o_active = torch.zeros(B, M, dtype=torch.bool, device=dev)
        self.o_r1 = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.o_uav = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.o_s1 = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.o_s2 = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.o_r2 = torch.full((B, M), -1, dtype=torch.long, device=dev)
        self.o_timedout = torch.zeros(B, M, dtype=torch.bool, device=dev)
        self.o_count = torch.zeros(B, dtype=torch.long, device=dev)

        # Per-step counters
        self.delivered = torch.zeros(B, dtype=torch.long, device=dev)
        self.uav_launched = torch.zeros(B, dtype=torch.long, device=dev)
        self.uav_balance = torch.zeros(B, dtype=torch.float32, device=dev)
        self.handoff = torch.zeros(B, dtype=torch.long, device=dev)
        self.handoff_opt = torch.zeros(B, dtype=torch.long, device=dev)

        # Episode tracking (for Monitor-like info)
        self._ep_rew = torch.zeros(B, dtype=torch.float32, device=dev)
        self._ep_len = torch.zeros(B, dtype=torch.long, device=dev)
        self._ep_start = np.full(B, _time.time())

        # Reward component storage (for tensorboard callback)
        self._last_reward_comps: dict[str, torch.Tensor] | None = None

        # Pre-allocated constants (avoid per-step tensor creation)
        self._big = torch.tensor(1e9, device=dev)
        self._neg1_f = torch.tensor(-1.0, device=dev)
        self._zero_f = torch.tensor(0.0, device=dev)
        self._penalty = torch.tensor(-200.0, device=dev)
        self._uav_stop_t = torch.tensor(UAV_STOP, dtype=torch.long, device=dev)
        # Station exclusion mask for demand group computation: (S, S)
        self._st_exclude = torch.eye(S, device=dev) * 1e9

        torch.manual_seed(seed)

        self._actions: np.ndarray | None = None
        self._reset_all()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
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

        # Riders
        self.r_pos[idx] = torch.randint(
            0, self.grid + 1, (n, self.R, 2), device=dev, dtype=torch.float32
        )
        self.r_free[idx] = True
        self.r_carrying[idx] = -1
        self.r_pending[idx] = -1
        self.r_target[idx] = 0
        self.r_has_tgt[idx] = False
        self.r_move_buf[idx] = 0

        # UAVs – initial station = uid % n_stations
        uav_init_sid = torch.arange(self.U, device=dev) % self.S  # (U,)
        self.u_pos[idx] = self.station_pos[uav_init_sid].unsqueeze(0).expand(n, -1, -1)
        self.u_bat[idx] = 1.0
        self.u_station[idx] = uav_init_sid.unsqueeze(0).expand(n, -1)
        self.u_tgt[idx] = -1
        self.u_state[idx] = UAV_STOP
        self.u_orders[idx] = -1
        self.u_ord_cnt[idx] = 0

        # Orders
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

        # Episode tracking
        self._ep_rew[idx] = 0
        self._ep_len[idx] = 0
        now = _time.time()
        for i in idx.cpu().tolist():
            self._ep_start[i] = now

    def reset(self) -> np.ndarray:
        self._reset_all()
        return self._build_obs().cpu().numpy()

    # ------------------------------------------------------------------
    # Step (VecEnv interface)
    # ------------------------------------------------------------------
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

        # Reward components for tensorboard callback
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

        # Episode tracking
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
                if trunc_np[i] and not term_np[i]:
                    infos[i]["TimeLimit.truncated"] = True
                else:
                    infos[i]["TimeLimit.truncated"] = False

            self._reset_envs(dones)
            reset_obs = self._build_obs().cpu().numpy()
            obs_np[done_np] = reset_obs[done_np]

            self._ep_rew[done_idx] = 0
            self._ep_len[done_idx] = 0
            now = _time.time()
            for i in done_idx.cpu().tolist():
                self._ep_start[i] = now

        return obs_np, rew_np, done_np, infos

    # ------------------------------------------------------------------
    # Core vectorised step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _step_impl(self, actions: torch.Tensor):
        B = self.num_envs
        dev = self.device

        rider_act = actions[:, : self.R]  # (B, R)
        station_act = actions[:, self.R :]  # (B, S)

        self.time_t += 1

        # Reset per-step counters
        self.delivered.zero_()
        self.uav_launched.zero_()
        self.uav_balance.zero_()
        self.handoff.zero_()
        self.handoff_opt.zero_()

        # 1. Update order wait times
        self.o_twait.add_(self.o_active.long())

        # 2-6. Sub-steps
        self._generate_orders()
        self._process_station_actions(station_act)
        self._process_rider_actions(rider_act)
        self._move_riders()
        self._move_uavs()
        self._charge_uavs()
        self._handle_rider_last_mile()

        # 7. Observations & rewards
        obs = self._build_obs()
        rewards, comps = self._compute_rewards()
        self._last_reward_comps = comps

        terminated = self.time_t >= self.max_steps_val
        n_active = self.o_active.sum(dim=-1)
        truncated = n_active > 100

        rewards = torch.where(truncated, self._penalty, rewards)
        dones = terminated | truncated
        return obs, rewards, terminated, truncated, dones

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_orders(self):
        B, R, dev = self.num_envs, self.R, self.device

        should_gen = torch.rand(B, device=dev) < 0.1
        free = self.r_carrying < 0  # (B, R)
        has_free = free.any(dim=-1)
        gen = should_gen & has_free & (self.o_count < self.max_orders)
        if not gen.any():
            return

        shop_idx = torch.randint(self.n_shops, (B,), device=dev)
        start = self.shop_pos[shop_idx]  # (B, 2)
        end = torch.randint(0, self.grid + 1, (B, 2), device=dev, dtype=torch.float32)

        dist = _manhattan(self.r_pos, start.unsqueeze(1))  # (B, R)
        dist = torch.where(free, dist, self._big)
        closest = dist.argmin(dim=-1)  # (B,)

        ei = torch.where(gen)[0]
        slots = self.o_count[gen]
        ri = closest[gen]

        self.o_start[ei, slots] = start[gen]
        self.o_end[ei, slots] = end[gen]
        self.o_status[ei, slots] = UNASSIGNED
        self.o_tcreated[ei, slots] = self.time_t[gen]
        self.o_twait[ei, slots] = 0
        self.o_active[ei, slots] = True
        self.o_r1[ei, slots] = ri
        self.o_uav[ei, slots] = -1
        self.o_s1[ei, slots] = -1
        self.o_s2[ei, slots] = -1
        self.o_r2[ei, slots] = -1
        self.o_timedout[ei, slots] = False

        self.r_free[ei, ri] = False
        self.r_carrying[ei, ri] = slots
        self.r_target[ei, ri] = start[gen]
        self.r_has_tgt[ei, ri] = True

        self.o_count[gen] += 1

    # ------------------------------------------------------------------
    # Station actions (loop over S stations, vectorised over B envs)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _process_station_actions(self, acts: torch.Tensor):
        B, S, dev = self.num_envs, self.S, self.device
        TIMEOUT = 20

        for sid in range(S):
            act_s = acts[:, sid]  # (B,)
            tidx = act_s - 1  # target station index per env

            valid = (tidx >= 0) & (tidx != sid) & (tidx < S)
            if not valid.any():
                continue

            # Timeout fallback (only for envs with valid action)
            waiting_all = (
                (self.o_status == AT_STATION)
                & (self.o_s1 == sid)
                & self.o_active
                & ~self.o_timedout
            )
            timed = waiting_all & (self.o_twait > TIMEOUT) & valid.unsqueeze(1)
            self.o_timedout |= timed

            # Re-derive waiting after timeout
            waiting = (
                (self.o_status == AT_STATION)
                & (self.o_s1 == sid)
                & self.o_active
                & ~self.o_timedout
            )

            # UAV availability
            uav_at = (self.u_station == sid) & (self.u_state != UAV_FLYING)  # (B, U)
            has_uav = uav_at.any(dim=-1)

            # Candidates: target station closer to order end than current
            cur_pos = self.station_pos[sid]  # (2,)
            tgt_pos = self.station_pos[tidx.clamp(0, S - 1)]  # (B, 2)

            cur_dist = _manhattan(
                cur_pos.unsqueeze(0).unsqueeze(0), self.o_end
            )  # (B, M)
            tgt_dist = _manhattan(tgt_pos.unsqueeze(1), self.o_end)  # (B, M)
            candidates = waiting & (tgt_dist < cur_dist)
            has_cand = candidates.any(dim=-1)

            # Balance logic
            tgt_uav_at = (self.u_station == tidx.clamp(0, S - 1).unsqueeze(1)) & (
                self.u_state != UAV_FLYING
            )
            tgt_uav_cnt = tgt_uav_at.sum(dim=-1).float()

            tgt_wait = (
                (self.o_status == AT_STATION)
                & (self.o_s1 == tidx.clamp(0, S - 1).unsqueeze(1))
                & self.o_active
                & ~self.o_timedout
            )
            tgt_wait_cnt = tgt_wait.sum(dim=-1).float()

            tgt_needs = (tgt_uav_cnt < 1) | (
                tgt_wait_cnt > tgt_uav_cnt.clamp(min=1) * 2
            )
            src_uav_cnt = uav_at.sum(dim=-1).float()
            src_wait_cnt = waiting.sum(dim=-1).float()
            src_gives = (src_uav_cnt > 3) | (src_wait_cnt == 0)

            bal_trigger = ~has_cand & tgt_needs & src_gives
            should_try = valid & has_uav & (has_cand | bal_trigger)
            if not should_try.any():
                continue

            # Select best-battery UAV
            bat_m = torch.where(uav_at, self.u_bat, self._neg1_f)
            best_bat, best_uid = bat_m.max(dim=-1)

            # Battery check
            cur_f = self.station_pos[sid].float()
            tgt_f = tgt_pos.float()
            flight_dist = ((tgt_f - cur_f.unsqueeze(0)) ** 2).sum(-1).sqrt()
            cost = flight_dist * 0.001
            bat_ok = (best_bat - cost) >= 0.1
            should_launch = should_try & bat_ok
            if not should_launch.any():
                continue

            # Balance reward
            self.uav_balance += (should_launch & bal_trigger).float()

            # Select top-K candidates by wait time
            K = min(self.uav_cap, self.max_orders)
            score = torch.where(
                candidates & should_launch.unsqueeze(1),
                self.o_twait.float(),
                -self._big,
            )
            topk_vals, topk_ids = score.topk(K, dim=-1)
            topk_ok = topk_vals > (-self._big + 1)

            ei = torch.where(should_launch)[0]
            ui = best_uid[should_launch]

            # Clear UAV orders
            self.u_orders[ei, ui] = -1

            for k in range(K):
                vk = should_launch & topk_ok[:, k]
                if not vk.any():
                    continue
                ek = torch.where(vk)[0]
                ok = topk_ids[vk, k]
                uk = best_uid[vk]

                self.o_status[ek, ok] = IN_UAV
                self.o_uav[ek, ok] = uk
                self.u_orders[ek, uk, k] = ok

            self.u_ord_cnt[ei, ui] = topk_ok[should_launch].sum(dim=-1)
            self.u_station[ei, ui] = -1
            self.u_tgt[ei, ui] = tidx[should_launch]
            self.u_state[ei, ui] = UAV_FLYING

            self.uav_launched += should_launch.long()

    # ------------------------------------------------------------------
    # Rider actions (fully vectorised over B × R)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _process_rider_actions(self, acts: torch.Tensor):
        B, R, S = self.num_envs, self.R, self.S
        dev = self.device

        can_act = ~self.r_has_tgt & (self.r_pending < 0)
        has_order = self.r_carrying >= 0

        # Gather carried order info
        soid = self.r_carrying.clamp(min=0)
        c_status = torch.where(
            has_order,
            self.o_status.gather(1, soid),
            torch.full_like(soid, -1),
        )
        c_start = self.o_start.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))
        c_end = self.o_end.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))

        # Case 1: no order, action > 0 → go to station
        case1 = can_act & ~has_order & (acts > 0)
        sid1 = (acts - 1).clamp(0, S - 1)
        pos1 = self.station_pos[sid1]

        # Case 2: UNASSIGNED → go to order.start
        case2 = can_act & has_order & (c_status == UNASSIGNED)

        # Case 3: PICKED_BY_R2 (last mile) → go to order.end
        case3 = can_act & has_order & (c_status == PICKED_BY_R2)

        remaining = (
            can_act
            & has_order
            & (c_status != UNASSIGNED)
            & (c_status != PICKED_BY_R2)
            & (c_status >= 0)
        )

        # Distances rider → stations
        r2st = _manhattan(
            self.r_pos.unsqueeze(2),
            self.station_pos.unsqueeze(0).unsqueeze(0),
        )  # (B, R, S)
        min_st, best_sid = r2st.min(dim=-1)
        dist_end = _manhattan(self.r_pos, c_end)

        # Case 4: force station
        frand = torch.rand(B, R, device=dev) < self.force_prob
        case4 = remaining & frand & (dist_end > 2 * min_st)
        pos4 = self.station_pos[best_sid]
        case4_opt = case4 & ((acts - 1) == best_sid)

        remaining2 = remaining & ~case4

        # Case 5a: action == 0 → direct delivery
        case5a = remaining2 & (acts == 0)

        # Case 5b: action > 0 → go to station (with optimisation check)
        case5b = remaining2 & (acts > 0)
        sid5 = (acts - 1).clamp(0, S - 1)

        st2end = _manhattan(
            self.station_pos.unsqueeze(0).unsqueeze(0),
            c_end.unsqueeze(2),
        )  # (B, R, S)
        closest_dest = st2end.argmin(dim=-1)

        forbid = case5b & (sid5 == closest_dest)
        case5b_ok = case5b & ~forbid
        case5a = case5a | forbid
        pos5 = self.station_pos[sid5]

        case5b_opt = case5b_ok & (sid5 == best_sid)

        # Compose targets
        any_case = case1 | case2 | case3 | case4 | case5a | case5b_ok
        new_tgt = torch.zeros(B, R, 2, device=dev)
        new_tgt = torch.where(case1.unsqueeze(-1), pos1, new_tgt)
        new_tgt = torch.where(case2.unsqueeze(-1), c_start, new_tgt)
        new_tgt = torch.where(case3.unsqueeze(-1), c_end, new_tgt)
        new_tgt = torch.where(case4.unsqueeze(-1), pos4, new_tgt)
        new_tgt = torch.where(case5a.unsqueeze(-1), c_end, new_tgt)
        new_tgt = torch.where(case5b_ok.unsqueeze(-1), pos5, new_tgt)

        self.r_target = torch.where(any_case.unsqueeze(-1), new_tgt, self.r_target)
        self.r_has_tgt = self.r_has_tgt | any_case

        self.handoff_opt += (case4_opt | case5b_opt).long().sum(dim=-1)

    # ------------------------------------------------------------------
    # Rider movement
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _move_riders(self):
        dev = self.device

        # Phase 1: handle riders already at target
        pre_at = self.r_has_tgt & (self.r_pos == self.r_target).all(dim=-1)
        self._handle_rider_arrivals()

        # Phase 2: physical movement (exclude riders handled in phase 1)
        has_tgt = self.r_has_tgt
        at_tgt = has_tgt & (self.r_pos == self.r_target).all(dim=-1)
        needs = has_tgt & ~at_tgt & ~pre_at

        self.r_move_buf += needs.float() * self.rider_speed
        can_step = needs & (self.r_move_buf >= 1.0)

        if can_step.any():
            self.r_move_buf = torch.where(
                can_step, self.r_move_buf - 1.0, self.r_move_buf
            )
            diff = self.r_target - self.r_pos
            ad = diff.abs()
            px = ad[..., 0] > ad[..., 1]
            py = ~px & (ad[..., 1] > 0)

            sx = torch.where(can_step & px, diff[..., 0].sign(), torch.zeros_like(diff[..., 0]))
            sy = torch.where(can_step & py, diff[..., 1].sign(), torch.zeros_like(diff[..., 1]))
            self.r_pos = self.r_pos + torch.stack([sx, sy], dim=-1)

            arrived = can_step & (self.r_pos == self.r_target).all(dim=-1)
            self.r_move_buf = torch.where(
                arrived,
                torch.zeros_like(self.r_move_buf),
                self.r_move_buf,
            )

        # Phase 3: handle new arrivals
        self._handle_rider_arrivals()

    # ------------------------------------------------------------------
    # Rider arrival handling (vectorised)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _handle_rider_arrivals(self):
        B, R, S = self.num_envs, self.R, self.S
        dev = self.device

        at_tgt = self.r_has_tgt & (self.r_pos == self.r_target).all(dim=-1)
        if not at_tgt.any():
            return

        has_order = self.r_carrying >= 0
        has_pend = self.r_pending >= 0

        soid = self.r_carrying.clamp(min=0)
        c_status = torch.where(
            has_order,
            self.o_status.gather(1, soid),
            torch.full_like(soid, -1),
        )
        c_start = self.o_start.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))
        c_end = self.o_end.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))

        # Station check
        at_st = (
            self.r_pos.unsqueeze(2)
            == self.station_pos.unsqueeze(0).unsqueeze(0)
        ).all(dim=-1)  # (B, R, S)
        at_any_st = at_st.any(dim=-1)
        which_st = at_st.float().argmax(dim=-1)

        handled = torch.zeros(B, R, dtype=torch.bool, device=dev)

        # Case A1: no order, pending, at station → pick up pending
        a1 = at_tgt & ~has_order & has_pend & at_any_st & ~handled
        if a1.any():
            e, r = torch.where(a1)
            poid = self.r_pending[e, r]
            self.o_status[e, poid] = PICKED_BY_R2
            self.o_r2[e, poid] = r
            oend = self.o_end[e, poid]
            self.r_carrying[e, r] = poid
            self.r_pending[e, r] = -1
            self.r_target[e, r] = oend
            self.r_has_tgt[e, r] = True
            self.r_free[e, r] = False
            handled |= a1

        # Case A2: no order, no pending → free
        a2 = at_tgt & ~has_order & ~has_pend & ~handled
        if a2.any():
            self.r_has_tgt[a2] = False
            self.r_target[a2] = 0
            self.r_free[a2] = True
            handled |= a2

        # Case B: UNASSIGNED at order.start → pick up
        at_start = (self.r_pos == c_start).all(dim=-1)
        b = at_tgt & has_order & (c_status == UNASSIGNED) & at_start & ~handled
        if b.any():
            e, r = torch.where(b)
            oids = self.r_carrying[e, r]
            self.o_status[e, oids] = PICKED_BY_R1
            self.r_has_tgt[e, r] = False
            self.r_target[e, r] = 0
            self.r_free[e, r] = False
            handled |= b

        # Case C: at order.end → delivered
        at_end = (self.r_pos == c_end).all(dim=-1)
        c = at_tgt & has_order & at_end & ~handled
        if c.any():
            e, r = torch.where(c)
            oids = self.r_carrying[e, r]

            self.o_status[e, oids] = DELIVERED
            self.o_active[e, oids] = False
            self.delivered += c.long().sum(dim=-1)

            dt = self.time_t[e] - self.o_tcreated[e, oids]
            is_uav = self.o_uav[e, oids] >= 0

            _add = torch.zeros(B, dtype=torch.long, device=dev)
            _add.scatter_add_(0, e, torch.ones_like(e, dtype=torch.long))
            self.r_carrying[e, r] = -1
            self.r_has_tgt[e, r] = False
            self.r_target[e, r] = 0
            self.r_free[e, r] = True
            handled |= c

        # Case D: PICKED_BY_R1 at station → deposit
        d = at_tgt & has_order & (c_status == PICKED_BY_R1) & at_any_st & ~handled
        if d.any():
            e, r = torch.where(d)
            oids = self.r_carrying[e, r]
            st = which_st[e, r]

            self.o_s1[e, oids] = st
            self.o_status[e, oids] = AT_STATION
            self.r_carrying[e, r] = -1
            self.r_has_tgt[e, r] = False
            self.r_target[e, r] = 0
            self.r_free[e, r] = True
            self.handoff += d.long().sum(dim=-1)
            handled |= d

        # Case E: fallback – still carrying but no target (and not already handled)
        fb = (self.r_carrying >= 0) & ~self.r_has_tgt & ~handled
        if fb.any():
            e, r = torch.where(fb)
            oids = self.r_carrying[e, r]
            self.r_target[e, r] = self.o_end[e, oids]
            self.r_has_tgt[e, r] = True

    # ------------------------------------------------------------------
    # UAV movement
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _move_uavs(self):
        dev = self.device
        flying = (self.u_state == UAV_FLYING) & (self.u_tgt >= 0)
        if not flying.any():
            return

        tgt_sid = self.u_tgt.clamp(0, self.S - 1)
        tgt_pos = self.station_pos[tgt_sid].float()

        vec = tgt_pos - self.u_pos
        dist = (vec ** 2).sum(dim=-1).sqrt() + 1e-9

        self.u_bat = torch.where(
            flying,
            (self.u_bat - 0.001 * self.uav_speed).clamp(min=0),
            self.u_bat,
        )

        arrived = flying & (dist <= self.uav_speed)
        still = flying & ~arrived

        direction = vec / dist.unsqueeze(-1)
        new_pos = self.u_pos + direction * self.uav_speed

        self.u_pos = torch.where(
            arrived.unsqueeze(-1),
            tgt_pos,
            torch.where(still.unsqueeze(-1), new_pos, self.u_pos),
        )

        if arrived.any():
            self._handle_uav_arrivals(arrived)

    @torch.no_grad()
    def _handle_uav_arrivals(self, arrived: torch.Tensor):
        e, u = torch.where(arrived)
        if len(e) == 0:
            return

        tsid = self.u_tgt[e, u]

        self.u_station[e, u] = tsid
        self.u_state[e, u] = UAV_CHARGING
        self.u_tgt[e, u] = -1
        self.u_bat[e, u] = (self.u_bat[e, u] + 0.1).clamp(max=1.0)

        for k in range(self.uav_cap):
            ok = self.u_orders[e, u, k]
            valid = ok >= 0
            if not valid.any():
                continue
            ev = e[valid]
            ov = ok[valid]
            tv = tsid[valid]
            self.o_status[ev, ov] = AT_DROP_POINT
            self.o_s2[ev, ov] = tv

        self.u_orders[e, u] = -1
        self.u_ord_cnt[e, u] = 0

    # ------------------------------------------------------------------
    # UAV charging
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _charge_uavs(self):
        charging = self.u_state == UAV_CHARGING
        self.u_bat = torch.where(
            charging, (self.u_bat + 0.1).clamp(max=1.0), self.u_bat
        )
        full = charging & (self.u_bat >= 1.0)
        self.u_state = torch.where(full, self._uav_stop_t, self.u_state)

    # ------------------------------------------------------------------
    # Rider last-mile assignment (loop S stations, vectorised over B)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _handle_rider_last_mile(self):
        B, S, dev = self.num_envs, self.S, self.device

        for sid in range(S):
            st_pos = self.station_pos[sid]

            uav_del = (
                (self.o_status == AT_DROP_POINT)
                & (self.o_s2 == sid)
                & self.o_active
                & (self.o_r2 < 0)
            )
            to_del = (
                (self.o_status == AT_STATION)
                & (self.o_s1 == sid)
                & self.o_active
                & self.o_timedout
                & (self.o_r2 < 0)
            )
            orders_pool = uav_del | to_del
            has_ord = orders_pool.any(dim=-1)

            truly_free = (self.r_carrying < 0) & (self.r_pending < 0)
            has_free = truly_free.any(dim=-1)

            proc = has_ord & has_free
            if not proc.any():
                continue

            d2st = _manhattan(self.r_pos, st_pos.unsqueeze(0).unsqueeze(0))
            d2st = torch.where(truly_free, d2st, self._big)
            crid = d2st.argmin(dim=-1)

            wscore = torch.where(orders_pool, self.o_twait.float(), self._neg1_f)
            boid = wscore.argmax(dim=-1)

            ei = torch.where(proc)[0]
            ri = crid[proc]
            oi = boid[proc]

            self.r_pending[ei, ri] = oi
            self.r_free[ei, ri] = False
            self.r_target[ei, ri] = st_pos
            self.r_has_tgt[ei, ri] = True
            self.o_r2[ei, oi] = ri

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_obs(self) -> torch.Tensor:
        B, R, S = self.num_envs, self.R, self.S
        dev = self.device
        grid = max(1, self.grid)

        # --- Pre-compute station stats ---
        # UAV availability per station
        uav_avail = self.u_state != UAV_FLYING  # (B, U)
        us_safe = self.u_station.clamp(0, S - 1)
        u_oh = F.one_hot(us_safe, S).float() * uav_avail.unsqueeze(-1).float()
        uav_cnt = u_oh.sum(dim=1)  # (B, S)

        # Orders waiting per station
        w_mask = (
            (self.o_status == AT_STATION)
            & self.o_active
            & ~self.o_timedout
        )
        os1_safe = self.o_s1.clamp(0, S - 1)
        o_oh = F.one_hot(os1_safe, S).float() * w_mask.unsqueeze(-1).float()
        cong = o_oh.sum(dim=1)  # (B, S)

        # Max battery per station
        bat_exp = self.u_bat.unsqueeze(-1) * u_oh  # (B, U, S)
        max_bat = bat_exp.max(dim=1).values.clamp(min=0)  # (B, S)

        MAX_BUF = float(max(1, self.cfg.station_max_order_buffer))
        MAX_UAVS = float(max(1, self.cfg.station_max_uavs))
        MAX_DIST = 2.0 * grid

        # --- Rider observations ---
        rx = self.r_pos[..., 0] / grid
        ry = self.r_pos[..., 1] / grid
        tn = (self.time_t.float() / max(1, self.max_steps_val)).unsqueeze(1).expand(B, R)
        ho = (self.r_carrying >= 0).float()

        soid = self.r_carrying.clamp(min=0)
        c_wait = self.o_twait.gather(1, soid).float()
        c_end = self.o_end.gather(1, soid.unsqueeze(-1).expand(-1, -1, 2))
        w_norm = (c_wait / 60.0).clamp(max=1.0) * ho
        d_dest = _manhattan(self.r_pos, c_end).float() / MAX_DIST * ho

        r_base = torch.stack([rx, ry, ho, w_norm, d_dest, tn], dim=-1)  # (B, R, 6)

        r2s = _manhattan(
            self.r_pos.unsqueeze(2),
            self.station_pos.unsqueeze(0).unsqueeze(0),
        ).float() / MAX_DIST  # (B, R, S)

        cong_n = (cong / MAX_BUF).clamp(max=1.0).unsqueeze(1).expand(B, R, S)
        uav_n = (uav_cnt / MAX_UAVS).clamp(max=1.0).unsqueeze(1).expand(B, R, S)
        s_feat = torch.stack([r2s, cong_n, uav_n], dim=-1).reshape(B, R, S * 3)

        r_obs = torch.cat([r_base, s_feat], dim=-1).reshape(B, R * self._rider_obs_dim)

        # --- Station observations ---
        sx = self.station_pos[:, 0] / grid
        sy = self.station_pos[:, 1] / grid
        sxy = torch.stack([sx, sy], dim=-1).unsqueeze(0).expand(B, S, 2)
        tn_s = (self.time_t.float() / max(1, self.max_steps_val)).unsqueeze(1).expand(B, S)
        tw_n = (cong / MAX_BUF).clamp(max=1.0)
        uc_n = (uav_cnt / MAX_UAVS).clamp(max=1.0)

        s_base = torch.stack(
            [sxy[..., 0], sxy[..., 1], tn_s, tw_n, uc_n, max_bat], dim=-1
        )  # (B, S, 6)

        MAX_WAIT_EST = MAX_BUF * 60.0
        demand = torch.zeros(B, S, S * 2, device=dev)

        d2s_all = _manhattan(
            self.o_end.unsqueeze(2),
            self.station_pos.unsqueeze(0).unsqueeze(0),
        )  # (B, M, S)

        for sid in range(S):
            wmask_s = w_mask & (self.o_s1 == sid)
            if not wmask_s.any():
                continue
            d2s_ex = d2s_all + self._st_exclude[sid]
            tsid = d2s_ex.argmin(dim=-1)  # (B, M)

            t_oh = F.one_hot(tsid, S).float() * wmask_s.unsqueeze(-1).float()
            cnt_k = t_oh.sum(dim=1)  # (B, S)
            wt_k = (self.o_twait.float().unsqueeze(-1) * t_oh).sum(dim=1)

            demand[:, sid, 0::2] = (cnt_k / MAX_BUF).clamp(max=1.0)
            demand[:, sid, 1::2] = (wt_k / MAX_WAIT_EST).clamp(max=1.0)

        s_obs = torch.cat([s_base, demand], dim=-1).reshape(B, S * self._station_obs_dim)

        return torch.cat([r_obs, s_obs], dim=-1)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, S, dev = self.num_envs, self.S, self.device

        act_cnt = self.o_active.sum(dim=-1).float()
        active_order_term = -0.0001 * act_cnt
        delivered_term = 8.8 * self.delivered.float()

        tw = (self.o_twait.float() * self.o_active.float()).sum(dim=-1)
        total_wait_term = -0.0001 * tw

        ot = ((self.o_twait > 60) & self.o_active).sum(dim=-1).float()
        overtime_term = -0.2 * ot

        uav_launch_term = (-0.0001 + 0.5) * self.uav_launched.float()

        # Overflow
        uav_avail = self.u_state != UAV_FLYING
        us_safe = self.u_station.clamp(0, S - 1)
        u_oh = F.one_hot(us_safe, S).float() * uav_avail.unsqueeze(-1).float()
        ucnt = u_oh.sum(dim=1)  # (B, S)
        excess = (ucnt - self.cfg.station_max_uavs).clamp(min=0).sum(dim=-1)
        overflow_term = -0.1 * excess

        uav_balance_term = 0.5 * self.uav_balance
        handoff_term = 0.05 * self.handoff.float() * self.force_prob
        handoff_opt_term = 0.01 * self.handoff_opt.float() * self.force_prob

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
        }
        return reward, comps

    # ------------------------------------------------------------------
    # VecEnv interface boiler-plate
    # ------------------------------------------------------------------
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
