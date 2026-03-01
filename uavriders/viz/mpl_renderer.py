from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 


class MplRenderer:
    def __init__(self, figsize=(12, 9)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # Adjust layout to make room for legend on the right
        self.fig.subplots_adjust(right=0.75)

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

        # Add Legend
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', label='Station (站点)',
                   markerfacecolor='red', markeredgecolor='black', markersize=10),
        ]

        if hasattr(env, "shop_locs") and len(env.shop_locs) > 0:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Shop (商家)',
                                          markerfacecolor='#2ca02c', markeredgecolor='black', markersize=8))

        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', label='Rider (Free)\n(空闲骑手)',
                   markerfacecolor='none', markeredgecolor='#1f77b4', markersize=8, markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', label='Rider (Busy)\n(忙碌骑手)',
                   markerfacecolor='#1f77b4', markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker='D', color='w', label='UAV (无人机)',
                   markerfacecolor='#ff7f0e', markeredgecolor='black', markersize=8),
        ])

        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., title="Legend (图例)")

        return self.fig, self.ax

