from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EnvConfig:
    rider_speed: float = 0.5
    uav_speed: float = 1.5
    uav_capacity_limit: int = 5

    world_grid_size: int = 20
    station_locs: tuple[tuple[float, float], ...] = (
        (3, 3),
        (17, 17),
        (3, 17),
        (17, 3),
        (10.0, 10.0),
    )
    shop_locs: tuple[tuple[float, float], ...] = (
        (0, 2),
        (2, 0),
        (18, 20),
        (20, 18),
        (15, 3),
        (3, 15),
        (5, 17),
        (17, 5),
    )

    n_riders: int = 20
    n_uav_each_station: int = 5

    station_max_uavs: int = 10
    station_max_order_buffer: int = 50
    station_max_concurrent_launch: int = 3

    force_station_prob: float = 0.2

    @property
    def n_stations(self) -> int:
        return len(self.station_locs)

    @property
    def n_shops(self) -> int:
        return len(self.shop_locs)

    @property
    def n_uavs(self) -> int:
        return int(self.n_stations * self.n_uav_each_station)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_stations"] = self.n_stations
        data["n_uavs"] = self.n_uavs
        data["n_shops"] = self.n_shops
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvConfig":
        known_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        if "uav_speed" not in kwargs and "rider_speed" in kwargs:
            kwargs["uav_speed"] = float(kwargs["rider_speed"]) * 3.0
        if "station_locs" in kwargs:
            kwargs["station_locs"] = tuple(tuple(x) for x in kwargs["station_locs"])
        if "shop_locs" in kwargs:
            kwargs["shop_locs"] = tuple(tuple(x) for x in kwargs["shop_locs"])
        return cls(**kwargs)


def default_config() -> EnvConfig:
    return EnvConfig()
