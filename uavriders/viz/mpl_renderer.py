from __future__ import annotations

import matplotlib.pyplot as plt


class MplRenderer:
    def __init__(self, figsize=(9, 9)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def init_map(self, grid_size: int):
        self.ax.clear()
        for x in range(grid_size + 1):
            self.ax.plot([x, x], [0, grid_size], linewidth=0.5, color="#e0e0e0", zorder=0)
        for y in range(grid_size + 1):
            self.ax.plot([0, grid_size], [y, y], linewidth=0.5, color="#e0e0e0", zorder=0)

        self.ax.set_xlim(-1, grid_size + 1)
        self.ax.set_ylim(-1, grid_size + 1)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

    def render(self, env):
        self.init_map(int(env.grid_size))

        sx = [s.pos[0] for s in env.stations]
        sy = [s.pos[1] for s in env.stations]
        self.ax.scatter(sx, sy, c="red", marker="^", s=180, zorder=10, edgecolors="black")
        for s in env.stations:
            n_uav = len(s.uav_available)
            n_wait = len(s.orders_waiting)
            info_str = f"U:{n_uav}\nW:{n_wait}"
            text_color = "red" if n_wait > 5 else "#333333"
            self.ax.text(s.pos[0], s.pos[1] + 1.2, info_str, fontsize=9, ha="center", color=text_color, zorder=20)

        if hasattr(env, "shop_locs"):
            shop_x = [p[0] for p in env.shop_locs]
            shop_y = [p[1] for p in env.shop_locs]
            self.ax.scatter(shop_x, shop_y, c="#2ca02c", marker="s", s=120, zorder=9, edgecolors="black")

        # Render Riders: distinct style for carrying_order vs free
        # Free riders: hollow circle
        rx_free = [r.pos[0] for r in env.riders if r.carrying_order is None]
        ry_free = [r.pos[1] for r in env.riders if r.carrying_order is None]
        self.ax.scatter(rx_free, ry_free, facecolors="none", edgecolors="#1f77b4", marker="o", s=70, zorder=12, linewidth=1.5, label="Free Rider")

        # Busy riders: filled circle
        rx_busy = [r.pos[0] for r in env.riders if r.carrying_order is not None]
        ry_busy = [r.pos[1] for r in env.riders if r.carrying_order is not None]
        self.ax.scatter(rx_busy, ry_busy, c="#1f77b4", marker="o", s=70, zorder=12, edgecolors="black", label="Busy Rider")

        # Draw paths for riders with target_pos
        for r in env.riders:
            if r.target_pos is not None:
                self.ax.plot([r.pos[0], r.target_pos[0]], [r.pos[1], r.target_pos[1]], 
                             c="#1f77b4", linestyle="--", linewidth=1, alpha=0.5, zorder=11)

        ux = [u.pos[0] for u in env.uavs]
        uy = [u.pos[1] for u in env.uavs]
        self.ax.scatter(ux, uy, c="#ff7f0e", marker="D", s=60, zorder=11, edgecolors="black")

        # Draw paths for flying UAVs
        for u in env.uavs:
            if u.station_id is None and u.target_station is not None:
                target_pos = env.stations[u.target_station].pos
                self.ax.plot([u.pos[0], target_pos[0]], [u.pos[1], target_pos[1]], 
                             c="#ff7f0e", linestyle=":", linewidth=1.5, alpha=0.6, zorder=10)

        self.ax.set_title(f"t={getattr(env, 'time', 0)}")
        return self.fig, self.ax

