from __future__ import annotations

from dataclasses import dataclass, field

from uavriders.sim.entities import Order, Rider, Station, UAV


@dataclass(slots=True)
class EnvData:
    time: int = 0
    orders: list[Order] = field(default_factory=list)
    active_orders: list[Order] = field(default_factory=list)
    stations: list[Station] = field(default_factory=list)
    uavs: list[UAV] = field(default_factory=list)
    riders: list[Rider] = field(default_factory=list)


WorldState = EnvData

