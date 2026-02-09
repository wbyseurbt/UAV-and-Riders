import argparse
import os
import sys
from pathlib import Path

import matplotlib.animation as animation
from ray.rllib.policy.policy import Policy

sys.path.append(str(Path(__file__).resolve().parents[2]))

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv
from uavriders.viz.mpl_renderer import MplRenderer


class TrainedAgent:
    def __init__(self, checkpoint_dir: str):
        policy_path = os.path.join(checkpoint_dir, "policies", "default_policy")
        self.policy = Policy.from_checkpoint(policy_path)

    def compute_actions(self, obs):
        return self.policy.compute_single_action(obs)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--interval", type=int, default=200)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"Error: checkpoint 目录不存在: {args.checkpoint}")
        sys.exit(1)

    env = DeliveryUAVSingleAgentEnv(max_steps=args.max_steps, seed=args.seed)
    agent_model = TrainedAgent(args.checkpoint)
    renderer = MplRenderer()

    obs, _ = env.reset(seed=args.seed)

    def animate(_frame):
        nonlocal obs
        action = agent_model.compute_actions(obs)
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
