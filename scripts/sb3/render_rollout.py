import argparse
import os
import sys
from pathlib import Path

import matplotlib.animation as animation

from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).resolve().parents[2]))

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv
from uavriders.viz.mpl_renderer import MplRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--interval", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: 找不到模型文件: {args.model}")
        sys.exit(1)

    env = DeliveryUAVSingleAgentEnv(max_steps=args.max_steps, seed=args.seed)
    model = PPO.load(args.model)
    renderer = MplRenderer()

    obs, _ = env.reset(seed=args.seed)

    def animate(_frame):
        nonlocal obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        renderer.render(env)
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)
        return renderer.ax

    ani = animation.FuncAnimation(renderer.fig, animate, frames=args.frames, interval=args.interval, blit=False)
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
