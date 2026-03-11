"""GPU-native PPO that keeps obs/actions/rewards on GPU during rollout.

Eliminates the per-step CPU↔GPU data transfers that dominate wall time
when using TorchVecEnv with standard SB3 PPO.
"""
from __future__ import annotations

import time as _time

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance

from scripts.sb3.timing_utils import TimedPPO


# ======================================================================
# GPU-resident rollout buffer
# ======================================================================
class GpuRolloutBuffer:
    """Fixed-size rollout buffer stored entirely as GPU tensors."""

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = torch.zeros(buffer_size, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, n_envs, act_dim, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, n_envs, device=device)
        self.episode_starts = torch.zeros(buffer_size, n_envs, device=device)
        self.values = torch.zeros(buffer_size, n_envs, device=device)
        self.log_probs = torch.zeros(buffer_size, n_envs, device=device)
        self.advantages = torch.zeros(buffer_size, n_envs, device=device)
        self.returns = torch.zeros(buffer_size, n_envs, device=device)

        self.pos = 0
        self.full = False

    def reset(self):
        self.pos = 0
        self.full = False

    def add(self, obs, actions, rewards, episode_starts, values, log_probs):
        self.observations[self.pos] = obs
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.episode_starts[self.pos] = episode_starts
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        """GAE computation — runs entirely on GPU."""
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def get(self, batch_size: int | None = None):
        """Yield GPU-resident minibatches as ``RolloutBufferSamples``."""
        total = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = total

        indices = torch.randperm(total, device=self.device)

        obs_flat = self.observations.reshape(total, self.obs_dim)
        act_flat = self.actions.reshape(total, self.act_dim)
        val_flat = self.values.reshape(total)
        lp_flat = self.log_probs.reshape(total)
        adv_flat = self.advantages.reshape(total)
        ret_flat = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield RolloutBufferSamples(
                observations=obs_flat[idx],
                actions=act_flat[idx],
                old_values=val_flat[idx],
                old_log_prob=lp_flat[idx],
                advantages=adv_flat[idx],
                returns=ret_flat[idx],
            )


# ======================================================================
# GpuPPO — custom rollout + training loop
# ======================================================================
class GpuPPO(TimedPPO):
    """PPO variant that collects rollouts and trains on GPU without CPU transfers.

    Pass *gpu_env* (a ``TorchVecEnv``) for direct tensor access.  The same
    object should also be passed as *env* for SB3 bookkeeping.
    """

    def __init__(self, gpu_env, *args, **kwargs):
        self._gpu_env = gpu_env
        super().__init__(*args, **kwargs)

        obs_dim = gpu_env.observation_space.shape[0]
        act_dim = len(gpu_env.action_space.nvec)
        self._gpu_buf = GpuRolloutBuffer(
            buffer_size=self.n_steps,
            n_envs=gpu_env.num_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=torch.device(gpu_env.device),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        nvec = gpu_env.action_space.nvec
        assert len(set(nvec)) == 1, (
            f"Fused sampling requires equal nvec, got {nvec}"
        )
        self._n_actions = len(nvec)
        self._n_options = int(nvec[0])

        n = gpu_env.num_envs
        self._empty_infos: list[dict] = [{} for _ in range(n)]
        self._empty_dones = np.zeros(n, dtype=bool)

        self._last_obs_t: torch.Tensor | None = None
        self._last_ep_starts_t: torch.Tensor | None = None
        self._cb_locals: dict = {"self": self}

        if getattr(gpu_env, '_compile', True):
            try:
                self._fused_forward = torch.compile(
                    self._fused_forward,
                    fullgraph=False,
                )
                self._fused_evaluate = torch.compile(
                    self._fused_evaluate,
                    fullgraph=False,
                )
            except Exception as e:
                print(f"[GpuPPO] torch.compile not available: {e}")

    def _excluded_save_params(self) -> list[str]:
        """Prevent SB3 from trying to JSON-serialize GPU tensors / env / buffer."""
        excluded = super()._excluded_save_params()
        excluded.extend([
            "_gpu_env",
            "_gpu_buf",
            "_last_obs_t",
            "_last_ep_starts_t",
            "_empty_infos",
            "_empty_dones",
            "_cb_locals",
            "_fused_forward",
            "_fused_evaluate",
        ])
        return excluded

    # ------------------------------------------------------------------
    # CUDA warmup — prime allocator caches before real training
    # ------------------------------------------------------------------
    def _warmup_cuda(self, gpu):
        """Run dummy forward+step cycles to trigger torch.compile and prime CUDA."""
        compile_on = getattr(gpu, '_compile', False)
        if compile_on:
            print("[GpuPPO] Warming up CUDA + torch.compile (first run compiles kernels, may take 1-3 min) …")
        else:
            print("[GpuPPO] Warming up CUDA …")
        self.policy.set_training_mode(False)
        obs = gpu._build_obs()
        n_warmup = 40 if compile_on else 20
        for _ in range(n_warmup):
            actions, _, _ = self._fused_forward(obs)
            obs, _, _, _, _, _ = gpu.step_tensor(actions)
        torch.cuda.synchronize()
        gpu._reset_all()
        self._last_obs_t = gpu._build_obs()
        print("[GpuPPO] Warmup done.")

    # ------------------------------------------------------------------
    # Fused forward / evaluate — batched Categorical instead of Python loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _fused_forward(self, obs: torch.Tensor):
        """Policy forward with batched MultiDiscrete sampling.

        Replaces SB3's per-sub-action Python loop with a single
        Categorical over (B, n_actions, n_options).
        """
        feat = self.policy.extract_features(obs, self.policy.features_extractor)
        latent_pi, latent_vf = self.policy.mlp_extractor(feat)
        values = self.policy.value_net(latent_vf)

        logits = self.policy.action_net(latent_pi)  # (B, n_actions * n_options)
        logits = logits.reshape(-1, self._n_actions, self._n_options)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()                          # (B, n_actions)
        log_probs = dist.log_prob(actions).sum(dim=-1)   # (B,)
        return actions, values, log_probs

    def _fused_evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """evaluate_actions with batched Categorical."""
        feat = self.policy.extract_features(obs, self.policy.features_extractor)
        latent_pi, latent_vf = self.policy.mlp_extractor(feat)
        values = self.policy.value_net(latent_vf)

        logits = self.policy.action_net(latent_pi)
        logits = logits.reshape(-1, self._n_actions, self._n_options)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return values, log_probs, entropy

    # ------------------------------------------------------------------
    # Rollout collection — everything stays on GPU
    # ------------------------------------------------------------------
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.rollout_count += 1
        gpu = self._gpu_env
        n_envs = gpu.num_envs
        dev = gpu.device

        if self._last_obs_t is None:
            self._last_obs_t = gpu._build_obs()
            self._last_ep_starts_t = torch.ones(n_envs, device=dev)
            self._warmup_cuda(gpu)

        self._gpu_buf.reset()
        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        print(f"\n[Iteration {self.rollout_count}] Starting Rollout Collection...")
        t0 = _time.time()

        comp_sums: dict[str, torch.Tensor] = {}
        comp_count = 0

        for step in range(n_rollout_steps):
            actions, values, log_probs = self._fused_forward(self._last_obs_t)

            new_obs, rewards, dones, terminal_obs, trunc_mask, ep_info = gpu.step_tensor(actions)

            # Truncation bootstrapping (only runs when episodes end)
            if terminal_obs is not None:
                with torch.no_grad():
                    term_val = self.policy.predict_values(terminal_obs)
                rewards = rewards + self.gamma * term_val.squeeze(-1) * trunc_mask.float()

            self._gpu_buf.add(
                self._last_obs_t,
                actions,
                rewards,
                self._last_ep_starts_t,
                values.squeeze(-1),
                log_probs,
            )

            self._last_obs_t = new_obs
            self._last_ep_starts_t = dones.float()

            # Reward-component accumulation (GPU-only, one .sum per component)
            rc = gpu._last_reward_comps
            if rc is not None:
                for k, v in rc.items():
                    if k not in comp_sums:
                        comp_sums[k] = torch.tensor(0.0, device=dev)
                    comp_sums[k] += v.sum()
                comp_count += n_envs

            # Episode info → ep_info_buffer (rare: only when episodes end)
            if ep_info is not None:
                idx_np, rews_np, lens_np = ep_info
                for j in range(len(idx_np)):
                    self.ep_info_buffer.extend(
                        [{"r": float(rews_np[j]), "l": int(lens_np[j]), "t": 0.0}]
                    )

            self.num_timesteps += n_envs

            # Lightweight callback (empty infos to skip per-step dict overhead)
            self._cb_locals["infos"] = self._empty_infos
            self._cb_locals["dones"] = self._empty_dones
            callback.update_locals(self._cb_locals)
            if not callback.on_step():
                return False

        # GAE
        with torch.no_grad():
            last_values = self.policy.predict_values(self._last_obs_t)
        self._gpu_buf.compute_returns_and_advantage(
            last_values.squeeze(-1), self._last_ep_starts_t
        )

        elapsed = _time.time() - t0
        print(f"[Timing] Total Data Collection Time: {elapsed:.4f} s")

        # Log reward components (single CPU transfer at rollout end)
        if comp_count > 0:
            for k, total in comp_sums.items():
                self.logger.record(f"reward_components/{k}", total.item() / comp_count)

        callback.update_locals(self._cb_locals)
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Training — reads directly from GPU buffer
    # ------------------------------------------------------------------
    def train(self):
        print(f"[Iteration {self.rollout_count}] Starting Training...")
        t0 = _time.time()

        self.policy.set_training_mode(True)

        clip_range = self.clip_range(self._current_progress_remaining)
        lr = self.lr_schedule(self._current_progress_remaining)
        for pg in self.policy.optimizer.param_groups:
            pg["lr"] = lr

        buf = self._gpu_buf

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        all_kl: list[float] = []

        for epoch in range(self.n_epochs):
            approx_kl_epoch: list[float] = []

            for batch in buf.get(self.batch_size):
                values, log_prob, entropy = self._fused_evaluate(
                    batch.observations, batch.actions
                )
                values = values.flatten()

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - batch.old_log_prob)

                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                value_loss = F.mse_loss(batch.returns, values)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_frac)
                approx_kl_epoch.append(approx_kl)

            mean_kl = np.mean(approx_kl_epoch)
            all_kl.append(mean_kl)
            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                if self.verbose >= 1:
                    print(f"Early stopping at epoch {epoch} due to reaching max kl: {mean_kl:.4f}")
                break

        self._n_updates += self.n_epochs

        vals_np = buf.values.reshape(-1).cpu().numpy()
        rets_np = buf.returns.reshape(-1).cpu().numpy()
        ev = explained_variance(vals_np, rets_np)

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_kl))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", ev)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/learning_rate", lr)
        if hasattr(self, "clip_range_vf") and self.clip_range_vf is not None:
            self.logger.record(
                "train/clip_range_vf",
                self.clip_range_vf(self._current_progress_remaining),
            )

        elapsed = _time.time() - t0
        print(f"[Timing] Network Training Time: {elapsed:.4f} s")
