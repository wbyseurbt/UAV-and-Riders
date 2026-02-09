# 运行方式
## 训练
### SB3（推荐，单智能体）
- `python train.py --timesteps 2000000 --max-steps 200 --seed 0 --algo ppo --save-every-steps 50000`
或
- `python scripts/sb3/train_ppo.py --timesteps 2000000 --max-steps 200 --seed 0 --algo ppo --save-every-steps 50000`

### Ray（可选备份）
- `python scripts/ray/train_ppo.py --iters 200 --max-steps 200 --seed 0 --algo ray_ppo --save-every-iters 10`

## TensorBoard（查看奖励分量）
- `tensorboard --logdir ./logs`
- 在 Scalars 里查看 `reward_components/*`

## 渲染回放
### SB3（模型 .zip）
从 `./logs/ppo/<训练时间>/final_model.zip` 加载：
- `python run.py --model ./logs/ppo/<训练时间>/final_model.zip --max-steps 200 --seed 0`
或
- `python scripts/sb3/render_rollout.py --model ./logs/ppo/<训练时间>/final_model.zip --max-steps 200 --seed 0`

### Ray（checkpoint 目录）
- `python scripts/ray/render_rollout.py --checkpoint <你的checkpoint目录> --max-steps 200 --seed 0`
