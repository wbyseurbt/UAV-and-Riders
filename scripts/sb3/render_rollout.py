"""Render rollout using TorchVecEnv (single env) and MplRenderer."""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.animation as animation

import torch
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO

from uavriders.envs.torch_vec_env import TorchVecEnv
from uavriders.envs.torch.render_view import wrap_torch_env_for_render
from uavriders.viz.mpl_renderer import MplRenderer


def _extract_run_time_and_iter(model_path: str) -> tuple[str, str]:
    p = Path(model_path)
    parts = list(p.parts)
    run_time = ""
    try:
        ppo_idx = parts.index("ppo")
        if ppo_idx + 1 < len(parts):
            run_time = parts[ppo_idx + 1]
    except ValueError:
        run_time = ""
    if not run_time:
        run_time = p.parent.name or "unknown"

    stem = p.stem
    if stem == "final_model":
        iter_tag = "final"
    elif stem.startswith("iter_"):
        suffix = stem[len("iter_") :]
        iter_tag = suffix if suffix else "iter"
    else:
        iter_tag = stem or "model"
    return run_time, iter_tag


def _auto_save_path(model_path: str, project_root: Path, ext: str = ".mp4") -> str:
    run_time, iter_tag = _extract_run_time_and_iter(model_path)
    save_dir = project_root / "video"
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_iter = "".join(ch for ch in str(iter_tag) if ch.isalnum() or ch in ("-", "_")) or "iter"
    safe_time = "".join(ch for ch in str(run_time) if ch.isalnum() or ch in ("-", "_")) or "unknown"
    ext = (ext or ".mp4").lower()
    if not ext.startswith("."):
        ext = "." + ext

    base = f"{safe_time}_iter{safe_iter}"
    existing = sorted(save_dir.glob(f"{base}_*{ext}"))
    next_idx = len(existing) + 1
    return str(save_dir / f"{base}_{next_idx:02d}{ext}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--save", nargs="?", const="auto", default="", type=str)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = "cuda"

    if not os.path.exists(args.model):
        print(f"Error: 找不到模型文件: {args.model}")
        sys.exit(1)

    if animation.writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
        except ImportError:
            pass

    save_path = str(args.save).strip()
    project_root = Path(__file__).resolve().parents[2]
    has_display = bool(os.environ.get("DISPLAY"))
    if save_path:
        if save_path == "auto":
            save_path = _auto_save_path(args.model, project_root=project_root, ext=".mp4")
        matplotlib.use("Agg", force=True)
    elif not has_display:
        print("Error: 当前环境没有可用的图形界面（DISPLAY 为空），请使用 --save 导出为 mp4/gif")
        sys.exit(2)

    torch_env = TorchVecEnv(
        num_envs=1,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        compile=False,  # Disable torch.compile for rendering to avoid MSVC requirement
    )
    env = wrap_torch_env_for_render(torch_env, env_index=0)

    model = PPO.load(args.model, device=args.device)
    renderer = MplRenderer()

    obs, _ = env.reset(seed=args.seed)

    fps = int(args.fps)
    if fps <= 0:
        fps = max(1, int(round(1000 / max(1, int(args.interval)))))
    duration = float(args.duration)
    frames = int(args.frames)
    if duration > 0:
        frames = max(1, int(round(duration * fps)))

    def animate(_frame):
        nonlocal obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Debug print every 10 frames
        if _frame % 10 == 0:
             # Access raw torch env via adapter
             raw_env = env._env
             idx = env._idx
             n_active = raw_env.o_active[idx].sum().item()
             n_carrying = (raw_env.r_carrying[idx] >= 0).sum().item()
             step_del = raw_env.delivered[idx].item()
             print(f"Step {_frame}: Active={n_active}, Carrying={n_carrying}, StepDel={step_del}, TotalDel={env._total_delivered}")

        renderer.render(env)
        if terminated or truncated:
            # Use accumulators directly from Adapter
            total = env._total_delivered
            uav_launches = env._total_uav_launched
            
            rider_only = env._stats_rider_only_count
            uav_assist = env._stats_uav_assist_count
            rider_time_sum = env._stats_rider_only_time_sum
            uav_time_sum = env._stats_uav_assist_time_sum
            
            avg_rider_time = rider_time_sum / rider_only if rider_only > 0 else 0
            avg_uav_time = uav_time_sum / uav_assist if uav_assist > 0 else 0

            print("\n" + "=" * 60)
            print(f"  EPISODE SUMMARY (Steps: {args.max_steps})")
            print("=" * 60)
            print(f"Total Delivered:      {total}")
            print(f"  - Rider Only:       {rider_only:<5} (Avg Time: {avg_rider_time:.2f})")
            print(f"  - UAV Assisted:     {uav_assist:<5} (Avg Time: {avg_uav_time:.2f})")
            print("-" * 60)
            print("=" * 60 + "\n")
            
            obs, _ = env.reset(seed=args.seed)
        return renderer.ax

    ani = animation.FuncAnimation(renderer.fig, animate, frames=frames, interval=args.interval, blit=False)
    import matplotlib.pyplot as plt

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        if ext == ".gif":
            from matplotlib.animation import PillowWriter
            ani.save(save_path, writer=PillowWriter(fps=fps), dpi=int(args.dpi))
        else:
            if ext == "":
                save_path = save_path + ".mp4"
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=fps)
                ani.save(save_path, writer=writer, dpi=int(args.dpi))
            else:
                print("Error: 导出 mp4 需要 ffmpeg。请安装 ffmpeg 或改用 --save xxx.gif")
                sys.exit(3)
    else:
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
