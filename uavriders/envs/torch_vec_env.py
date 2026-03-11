"""Re-export TorchVecEnv from torch package for backward compatibility."""
from uavriders.envs.torch.vec_env import TorchVecEnv

__all__ = ["TorchVecEnv"]
