from __future__ import annotations

import numpy as np


def manhattan(p, q) -> int:
    p = np.asarray(p, dtype=int)
    q = np.asarray(q, dtype=int)
    return int(abs(p[0] - q[0]) + abs(p[1] - q[1]))

