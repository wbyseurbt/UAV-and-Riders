"""Rider movement and arrival handling for torch vectorized environment.

Sync-free: all operations use gather/scatter/torch.where (3-arg) to avoid
GPU-CPU synchronization from .any() or torch.where(mask) (nonzero).
"""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import (
    AT_STATION,
    DELIVERED,
    PICKED_BY_R1,
    PICKED_BY_R2,
    UNASSIGNED,
)


def move_riders(env) -> None:
    """Move riders toward targets then handle arrivals (single call, sync-free)."""
    has_tgt = env.r_has_tgt
    at_tgt = has_tgt & (env.r_pos == env.r_target).all(dim=-1)
    needs = has_tgt & ~at_tgt

    env.r_move_buf += needs.float() * env.rider_speed
    can_step = needs & (env.r_move_buf >= 1.0)

    env.r_move_buf = torch.where(can_step, env.r_move_buf - 1.0, env.r_move_buf)
    diff = env.r_target - env.r_pos
    ad = diff.abs()
    px = ad[..., 0] > ad[..., 1]
    py = ~px & (ad[..., 1] > 0)

    sx = torch.where(can_step & px, diff[..., 0].sign(), torch.zeros_like(diff[..., 0]))
    sy = torch.where(can_step & py, diff[..., 1].sign(), torch.zeros_like(diff[..., 1]))
    env.r_pos = env.r_pos + torch.stack([sx, sy], dim=-1)

    arrived = can_step & (env.r_pos == env.r_target).all(dim=-1)
    env.r_move_buf = torch.where(arrived, torch.zeros_like(env.r_move_buf), env.r_move_buf)

    handle_rider_arrivals(env)


# ---------------------------------------------------------------------------
# helpers for sync-free scatter into order tensors
# ---------------------------------------------------------------------------

def _scatter_order_1d(o_tensor, oid_idx, mask, new_val):
    """Scatter new_val into o_tensor[b, oid_idx[b,r]] where mask[b,r]."""
    cur = o_tensor.gather(1, oid_idx)
    o_tensor.scatter_(1, oid_idx, torch.where(mask, new_val, cur))


def handle_rider_arrivals(env) -> None:
    """Handle riders that have reached their target (sync-free)."""
    B, R, S = env.num_envs, env.R, env.S
    dev = env.device
    NULL = env._null_oid

    at_tgt = env.r_has_tgt & (env.r_pos == env.r_target).all(dim=-1)

    has_order = env.r_carrying >= 0
    has_pend = env.r_pending >= 0

    carry_safe = torch.where(has_order, env.r_carrying, NULL)
    pend_safe = torch.where(has_pend, env.r_pending, NULL)

    c_status = env.o_status.gather(1, carry_safe)
    c_start = env.o_start.gather(1, carry_safe.unsqueeze(-1).expand(-1, -1, 2))
    c_end = env.o_end.gather(1, carry_safe.unsqueeze(-1).expand(-1, -1, 2))

    at_st = (
        env.r_pos.unsqueeze(2) == env.station_pos.unsqueeze(0).unsqueeze(0)
    ).all(dim=-1)
    at_any_st = at_st.any(dim=-1)
    which_st = at_st.float().argmax(dim=-1)

    handled = torch.zeros(B, R, dtype=torch.bool, device=dev)

    # Case A1: arrived, no order, has pending, at station → pick up pending
    a1 = at_tgt & ~has_order & has_pend & at_any_st & ~handled

    PBR2 = torch.tensor(PICKED_BY_R2, device=dev, dtype=env.o_status.dtype).expand(B, R)
    _scatter_order_1d(env.o_status, pend_safe, a1, PBR2)
    r_idx = torch.arange(R, device=dev).unsqueeze(0).expand(B, R)
    _scatter_order_1d(env.o_r2, pend_safe, a1, r_idx)

    pend_end = env.o_end.gather(1, pend_safe.unsqueeze(-1).expand(-1, -1, 2))

    env.r_carrying = torch.where(a1, env.r_pending, env.r_carrying)
    env.r_pending = torch.where(a1, -1, env.r_pending)
    env.r_target = torch.where(a1.unsqueeze(-1), pend_end, env.r_target)
    env.r_has_tgt = env.r_has_tgt | a1
    env.r_free = env.r_free & ~a1
    handled = handled | a1

    # Case A2: arrived, no order, no pending → become free
    a2 = at_tgt & ~has_order & ~has_pend & ~handled
    env.r_has_tgt = env.r_has_tgt & ~a2
    env.r_target = torch.where(a2.unsqueeze(-1), torch.zeros_like(env.r_target), env.r_target)
    env.r_free = env.r_free | a2
    handled = handled | a2

    # Case B: UNASSIGNED at order start → mark picked
    at_start = (env.r_pos == c_start).all(dim=-1)
    b_case = at_tgt & has_order & (c_status == UNASSIGNED) & at_start & ~handled

    PBR1 = torch.tensor(PICKED_BY_R1, device=dev, dtype=env.o_status.dtype).expand(B, R)
    _scatter_order_1d(env.o_status, carry_safe, b_case, PBR1)
    env.r_has_tgt = env.r_has_tgt & ~b_case
    env.r_target = torch.where(b_case.unsqueeze(-1), torch.zeros_like(env.r_target), env.r_target)
    env.r_free = env.r_free & ~b_case
    handled = handled | b_case

    # Case C: arrived at order end → delivered
    at_end = (env.r_pos == c_end).all(dim=-1)
    c_case = at_tgt & has_order & at_end & ~handled

    DEL_T = torch.tensor(DELIVERED, device=dev, dtype=env.o_status.dtype).expand(B, R)
    _scatter_order_1d(env.o_status, carry_safe, c_case, DEL_T)
    _scatter_order_1d(env.o_active, carry_safe, c_case,
                      torch.zeros(B, R, dtype=torch.bool, device=dev))
    env.delivered += c_case.long().sum(dim=-1)

    env.r_carrying = torch.where(c_case, -1, env.r_carrying)
    env.r_has_tgt = env.r_has_tgt & ~c_case
    env.r_target = torch.where(c_case.unsqueeze(-1), torch.zeros_like(env.r_target), env.r_target)
    env.r_free = env.r_free | c_case
    handled = handled | c_case

    # Case D: PICKED_BY_R1 arrived at station → deposit
    d_case = at_tgt & has_order & (c_status == PICKED_BY_R1) & at_any_st & ~handled

    _scatter_order_1d(env.o_s1, carry_safe, d_case, which_st)
    ATS_T = torch.tensor(AT_STATION, device=dev, dtype=env.o_status.dtype).expand(B, R)
    _scatter_order_1d(env.o_status, carry_safe, d_case, ATS_T)
    # Reset wait timer when arriving at station so timeout counts from arrival
    _scatter_order_1d(env.o_twait, carry_safe, d_case, torch.zeros(B, R, dtype=torch.long, device=dev))
    env.r_carrying = torch.where(d_case, -1, env.r_carrying)
    env.r_has_tgt = env.r_has_tgt & ~d_case
    env.r_target = torch.where(d_case.unsqueeze(-1), torch.zeros_like(env.r_target), env.r_target)
    env.r_free = env.r_free | d_case
    env.handoff += d_case.long().sum(dim=-1)
    handled = handled | d_case

    # Fallback: carrying but no target → set target to order end
    fb = (env.r_carrying >= 0) & ~env.r_has_tgt & ~handled
    fb_safe = torch.where(fb, env.r_carrying.clamp(min=0), NULL)
    fb_end = env.o_end.gather(1, fb_safe.unsqueeze(-1).expand(-1, -1, 2))
    env.r_target = torch.where(fb.unsqueeze(-1), fb_end, env.r_target)
    env.r_has_tgt = env.r_has_tgt | fb
