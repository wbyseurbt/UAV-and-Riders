from __future__ import annotations

from typing import Any, Mapping

from uavriders.envs.single_env import DeliveryUAVSingleAgentEnv


def make_single_gym_env(env_config: Mapping[str, Any]):
    max_steps = int(env_config.get("max_steps", 200))
    seed = env_config.get("seed", None)
    config = env_config.get("config", None)
    return DeliveryUAVSingleAgentEnv(max_steps=max_steps, seed=seed, config=config)


def get_single_spaces(max_steps: int = 200, seed: int = 0, config=None):
    env = DeliveryUAVSingleAgentEnv(max_steps=max_steps, seed=seed, config=config)
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()
    return obs_space, act_space
