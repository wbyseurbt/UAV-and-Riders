Rider_SPEED = 0.5               # 格/分钟

UAV_SPEED = Rider_SPEED * 3     # 格/分钟
UAV_capacity_limit = 5          # 无人机最大载客量

World_grid_size = 20                # 网格大小

World_locs_stations = [(5, 5), (15, 15), (5, 15)]  # 充电站位置
World_locs_shops = [(0, 0), (20, 20), (0, 20), (20, 0), (10, 10)]  # 商店位置

World_n_riders = 10                 # 骑手数量
World_n_uav_each_station = 5        # 每个站点无人机数量
World_n_stations = len(World_locs_stations)                 # 充电站数量
World_n_uavs = World_n_stations * World_n_uav_each_station  # 总无人机数量
World_n_shops = len(World_locs_shops)                       # 商店数量

Station_MAX_UAVS = 10          # 站点最大无人机容量
Station_MAX_ORDER_BUFFER = 100      # 充电速率，单位：电

Station_MAX_CONCURRENT_LAUNCH = 3  # 最大同时起飞无人机数量