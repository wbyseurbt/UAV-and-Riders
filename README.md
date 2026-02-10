# 环境安装
最少依赖（训练 + TensorBoard）：

```bash
pip install -U stable-baselines3 tensorboard gymnasium
```

渲染回放需要 matplotlib：

```bash
pip install -U matplotlib
```

# 运行方式
## 训练（SB3 / PPO 单智能体）
推荐用 `train.py`（只是一个入口包装，参数与 `scripts/sb3/train_ppo.py` 完全一致）：

```bash
python train.py --iters 100 --max_steps 200 --seed 0
```

也可以直接运行脚本：

```bash
python scripts/sb3/train_ppo.py --iters 100 --max_steps 200 --seed 0
```

### 训练步数怎么控制？
本项目用 `--iters` 控制训练迭代次数，每次迭代的采样步数为：

```
每迭代步数 = n_steps * n_envs
总步数 = iters * n_steps * n_envs
```

其中：
- `n_steps`：PPO 每个环境每次 rollout 的步数（默认 2048）
- `n_envs`：并行环境数量（默认自动取 CPU 核心数；见下文“环境并行”）

如果你想大致对齐“总 timesteps”，可以自己换算出 iters。例如 `n_steps=2048, n_envs=16` 时，每迭代采样 `32768` 步。

### 环境并行（默认开启）
默认使用多进程并行环境（`SubprocVecEnv`）：
- `--vec_env` 默认是 `subproc`
- `--n_envs` 默认是 `0`，表示自动取 `CPU 核心数`

常用示例（显式指定 16 个并行环境）：

```bash
python train.py --n_envs 16
```

多进程启动方式（Linux 推荐 `forkserver`，默认也是它）：

```bash
python train.py --vec_env subproc --start_method forkserver
```

### policy 选哪个？
当前环境观测是 1D 向量（Box），推荐：
- `--policy MlpPolicy`（默认）

### 用 GPU 训练（4090）
强制把网络放到 GPU 上训练：

```bash
python train.py --device cuda
```

说明：
- PPO + `MlpPolicy` 通常瓶颈在环境 step 与小网络计算，GPU 利用率可能不高（SB3 会提示这一点）
- 脚本的 `--device auto` 在 `MlpPolicy` 时会默认选择 CPU；想用 GPU 就显式 `--device cuda`

### 想让 GPU 更“忙”一点
可以加大网络与每次更新的计算量（示例）：

```bash
python train.py --device cuda --n_envs 16 --n_steps 4096 --net_arch 512,512 --batch_size 0
```

可调 PPO 关键参数（完整参数列表见 `python train.py --help`）：
- `--n_steps`、`--batch_size`（`0` 表示自动匹配 `n_steps*n_envs`）、`--n_epochs`
- `--learning_rate`、`--gamma`、`--gae_lambda`、`--clip_range`
- `--ent_coef`、`--vf_coef`、`--max_grad_norm`
- `--net_arch`（例如 `256,256`）

### 训练输出与模型文件
训练日志目录默认是 `./logs`，最终结构类似：
- `./logs/ppo/<run_time>/tb_iter/`：TensorBoard events（横轴 step=iteration）
- `./logs/ppo/<run_time>/final_model.zip`：最终模型
- `./logs/ppo/<run_time>/checkpoints/iter_XXXX.zip`：按迭代保存的 checkpoint（由 `--save_every_iters` 控制）

## TensorBoard（查看奖励分量）

```bash
tensorboard --logdir ./logs
```

在 Scalars 里查看：
- `reward_components/*`

## 渲染回放（SB3 模型 .zip）
从 `./logs/ppo/<训练时间>/final_model.zip` 加载：

```bash
python run.py --model ./logs/ppo/<训练时间>/final_model.zip --max-steps 200 --seed 0
```

或：

```bash
python scripts/sb3/render_rollout.py --model ./logs/ppo/<训练时间>/final_model.zip --max-steps 200 --seed 0
```
