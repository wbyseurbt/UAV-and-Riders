"""Order generation for torch vectorized environment (sync-free)."""
from __future__ import annotations

import torch

from uavriders.envs.torch.constants import UNASSIGNED, manhattan


def generate_orders(env) -> None:
    """Generate new orders (vectorised over envs, sync-free)."""
    B, R, dev = env.num_envs, env.R, env.device

    should_gen = torch.rand(B, device=dev) < 0.1
    free = env.r_carrying < 0  # (B, R)
    has_free = free.any(dim=-1)  # per-env reduction, no global sync
    gen = should_gen & has_free & (env.o_count < env.max_orders)

    shop_idx = torch.randint(env.n_shops, (B,), device=dev)
    start = env.shop_pos[shop_idx]  # (B, 2)
    end = torch.randint(0, env.grid + 1, (B, 2), device=dev, dtype=torch.float32)

    dist = manhattan(env.r_pos, start.unsqueeze(1))  # (B, R)
    dist = torch.where(free, dist, env._big)
    closest = dist.argmin(dim=-1)  # (B,)

    slots = env.o_count  # (B,)
    slot_1d = slots.unsqueeze(1)  # (B, 1)
    slot_2d = slot_1d.unsqueeze(2).expand(-1, 1, 2)  # (B, 1, 2)
    gen_1d = gen.unsqueeze(1)  # (B, 1)
    gen_2d = gen_1d.unsqueeze(2).expand(-1, 1, 2)  # (B, 1, 2)

    # Scatter into order tensors at slot position (sync-free)
    def _scat1(tensor, val):
        cur = tensor.gather(1, slot_1d)
        tensor.scatter_(1, slot_1d, torch.where(gen_1d, val, cur))

    def _scat2(tensor, val):
        cur = tensor.gather(1, slot_2d)
        tensor.scatter_(1, slot_2d, torch.where(gen_2d, val, cur))

    _scat2(env.o_start, start.unsqueeze(1))
    _scat2(env.o_end, end.unsqueeze(1))
    _scat1(env.o_status, torch.full_like(slot_1d, UNASSIGNED))
    _scat1(env.o_tcreated, env.time_t.unsqueeze(1).expand_as(slot_1d))
    _scat1(env.o_twait, torch.zeros_like(slot_1d))
    _scat1(env.o_active, torch.ones(B, 1, dtype=torch.bool, device=dev))
    _scat1(env.o_r1, closest.unsqueeze(1))
    _scat1(env.o_uav, torch.full_like(slot_1d, -1))
    _scat1(env.o_s1, torch.full_like(slot_1d, -1))
    _scat1(env.o_s2, torch.full_like(slot_1d, -1))
    _scat1(env.o_r2, torch.full_like(slot_1d, -1))
    _scat1(env.o_timedout, torch.zeros(B, 1, dtype=torch.bool, device=dev))

    # Scatter into rider tensors (sync-free)
    ri_1d = closest.unsqueeze(1)  # (B, 1)
    ri_2d = ri_1d.unsqueeze(2).expand(-1, 1, 2)  # (B, 1, 2)

    cur_free = env.r_free.gather(1, ri_1d)
    env.r_free.scatter_(1, ri_1d, torch.where(gen_1d, False, cur_free))

    cur_carry = env.r_carrying.gather(1, ri_1d)
    env.r_carrying.scatter_(1, ri_1d, torch.where(gen_1d, slots.unsqueeze(1), cur_carry))

    cur_tgt = env.r_target.gather(1, ri_2d)
    env.r_target.scatter_(1, ri_2d, torch.where(gen_2d, start.unsqueeze(1), cur_tgt))

    cur_htgt = env.r_has_tgt.gather(1, ri_1d)
    env.r_has_tgt.scatter_(1, ri_1d, torch.where(gen_1d, True, cur_htgt))

    env.o_count += gen.long()
