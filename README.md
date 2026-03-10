# UAV-and-Riders

基于强化学习的无人机-骑手协同配送系统。在 20×20 网格世界中，RL 智能体学习调度 30 个骑手和 25 架无人机（分布在 5 个中转站），完成从商铺到终点的订单配送。

## 快速开始

### 安装

```bash
conda create -n RL python=3.10 -y
conda activate RL
pip install gymnasium numpy "stable-baselines3[extra]" matplotlib tensorboard torch
```

### 训练

```bash
python train.py --iters 100
```

### 渲染回放

```bash
python run.py --model ./logs/ppo/<run_time>/final_model.zip
```

## 训练模式

### CPU 多进程（默认）

使用 `SubprocVecEnv`，每个环境独立进程：

```bash
python train.py --n_envs 16
```

### GPU 向量化

使用 `TorchVecEnv` + `GpuPPO`，所有环境在 GPU 上批量运算，数据全程驻留显存：

```bash
python train.py --vec_env torch --device cuda --n_envs 256 --n_steps 1024
```

> 相比 CPU 16 进程，GPU 256 环境可达 **6.5x** 端到端加速。

## PPO 参数说明

### 训练流程

```
n_envs 个并行环境 → 每个采样 n_steps 步 → 得到 rollout (n_envs × n_steps 条数据)
→ 切成 batch_size 大小的 mini-batch → 训练 n_epochs 轮 → 更新策略 → 重复 iters 次
```

总训练步数 = `iters × n_envs × n_steps`

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--iters` | 100 | 训练迭代次数 |
| `--n_envs` | 0 (=CPU核数) | 并行环境数量 |
| `--n_steps` | 2048 | 每环境每次 rollout 的步数 |
| `--max_steps` | 1024 | 单个 episode 最大步数 |
| `--batch_size` | 0 (=全量) | mini-batch 大小；`0` 表示 `n_envs × n_steps` |
| `--n_epochs` | 10 | 每次 rollout 数据的训练轮数 |
| `--learning_rate` | 3e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子，越大越重视长期收益 |
| `--gae_lambda` | 0.95 | GAE 平滑系数，平衡偏差和方差 |
| `--clip_range` | 0.2 | PPO 策略裁剪范围，限制单次更新幅度 |
| `--ent_coef` | 0.0 | 熵系数，越大鼓励更多探索 |
| `--vf_coef` | 0.5 | 价值函数损失权重 |
| `--max_grad_norm` | 0.5 | 梯度裁剪阈值，防止梯度爆炸 |
| `--net_arch` | "" (=64,64) | 网络隐藏层，如 `256,256` |
| `--vec_env` | subproc | 环境类型：`dummy` / `subproc` / `torch` |
| `--device` | auto | 训练设备：`cpu` / `cuda` / `auto` |

### 参数调优建议

- **加快采样**：增大 `--n_envs`（CPU 用 16-32，GPU 用 128-512）
- **稳定训练**：减小 `--learning_rate`（如 1e-4），增大 `--n_steps`
- **鼓励探索**：增大 `--ent_coef`（如 0.01）
- **增强网络**：`--net_arch 256,256` 或 `512,512`

## 输出结构

```
logs/ppo/<run_time>/
├── final_model.zip              # 最终模型
├── checkpoints/iter_XXXX.zip    # 每 save_every_iters 轮保存
└── tb_iter/                     # TensorBoard 日志
```

查看训练曲线：

```bash
tensorboard --logdir ./logs/ppo/<run_time>/
```

主要指标：`reward_components/*`（各奖励分量）、`rollout/ep_rew_mean`（回合平均奖励）

## 渲染回放

```bash
# 窗口播放
python run.py --model ./logs/ppo/<run_time>/final_model.zip

# 导出视频
python scripts/sb3/render_rollout.py --model ./logs/ppo/<run_time>/final_model.zip --save

# 指定时长和帧率
python scripts/sb3/render_rollout.py --model <model_path> --save --duration 30 --fps 30
```

视频自动保存到 `./video/`，命名格式：`<run_time>_iter<N>_<序号>.mp4`

## 项目结构

```
├── train.py                     # 训练入口
├── run.py                       # 渲染入口
├── uavriders/
│   ├── configs/env_config.py    # 环境配置（地图、速度、数量等）
│   ├── envs/
│   │   ├── single_env.py        # Gymnasium 单环境
│   │   └── torch_vec_env.py     # GPU 批量向量化环境
│   ├── sim/                     # 仿真逻辑（骑手、无人机、订单、站点）
│   ├── rl/                      # 观测构建、奖励计算
│   └── viz/                     # matplotlib 可视化
└── scripts/sb3/
    ├── train_ppo.py             # PPO 训练主逻辑
    ├── gpu_ppo.py               # GPU 原生 PPO（消除 CPU↔GPU 传输）
    ├── render_rollout.py        # 回放渲染
    ├── timing_utils.py          # 训练计时工具
    ├── tensorboard.py           # TensorBoard 回调
    └── savemodel.py             # 模型保存回调
```
