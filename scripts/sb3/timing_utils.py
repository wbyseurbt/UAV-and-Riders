
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class TimingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_step_time = 0.0
        self.step_counts = 0
        
    def step(self, action):
        start = time.time()
        result = self.env.step(action)
        dt = time.time() - start
        self.total_step_time += dt
        self.step_counts += 1
        return result

    def get_time_stats(self):
        return self.total_step_time, self.step_counts
    
    def reset_time_stats(self):
        self.total_step_time = 0.0
        self.step_counts = 0

class TimedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_count = 0

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.rollout_count += 1
        print(f"\n[Iteration {self.rollout_count}] Starting Rollout Collection...")
        
        # Reset stats in wrappers if possible (only works for DummyVecEnv easily)
        if isinstance(env, DummyVecEnv):
            for i in range(len(env.envs)):
                # Handle nested wrappers
                current = env.envs[i]
                while hasattr(current, 'env'):
                    if isinstance(current, TimingWrapper):
                        current.reset_time_stats()
                        break
                    current = current.env

        start_time = time.time()
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        total_rollout_time = time.time() - start_time
        
        print(f"[Timing] Total Data Collection Time: {total_rollout_time:.4f} s")
        
        # Try to get simulation time
        sim_time = 0.0
        total_steps = 0
        
        if isinstance(env, DummyVecEnv):
            for i in range(len(env.envs)):
                current = env.envs[i]
                found = False
                while hasattr(current, 'env'):
                    if isinstance(current, TimingWrapper):
                        t, c = current.get_time_stats()
                        sim_time += t
                        total_steps += c
                        found = True
                        break
                    current = current.env
            
            if total_steps > 0:
                # Average simulation time per step (across all envs)
                # But since DummyVecEnv is sequential, sum is correct for total CPU time spent in env
                print(f"[Timing] -> Simulation (Env.step) Time: {sim_time:.4f} s ({sim_time/total_rollout_time*100:.1f}%)")
                print(f"[Timing] -> Inference & Overhead Time: {total_rollout_time - sim_time:.4f} s")
        
        elif isinstance(env, SubprocVecEnv):
            print(f"[Timing] (Detailed simulation time not available in SubprocVecEnv, switch to DummyVecEnv to see env.step time)")

        return result

    def train(self):
        print(f"[Iteration {self.rollout_count}] Starting Training...")
        start_time = time.time()
        super().train()
        train_time = time.time() - start_time
        print(f"[Timing] Network Training Time: {train_time:.4f} s")
