import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.animation as animation

from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).resolve().parents[2]))

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv
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
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: 找不到模型文件: {args.model}")
        sys.exit(1)

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

    env = DeliveryUAVSingleAgentEnv(max_steps=args.max_steps, seed=args.seed)
    model = PPO.load(args.model)
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
        renderer.render(env)
        if terminated or truncated:
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
