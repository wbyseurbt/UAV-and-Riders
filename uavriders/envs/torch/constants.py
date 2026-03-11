"""Constants and small utilities for torch vectorized environment."""
from __future__ import annotations

import torch

# Order status (must match semantics used in step logic)
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


def manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched manhattan distance along the last dimension (size 2)."""
    return (a - b).abs().sum(dim=-1)
