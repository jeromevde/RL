"""
Proximal Policy Optimization (PPO)
===================================
Paper: Schulman et al., 2017 — "Proximal Policy Optimization Algorithms"
       https://arxiv.org/abs/1707.06347

Paper (GAE): Schulman et al., 2016 — "High-Dimensional Continuous Control Using
       Generalised Advantage Estimation" — https://arxiv.org/abs/1506.02438

Environment: CartPole-v1 (discrete)   or   Pendulum-v1 (continuous)

Theory
------
PPO is an on-policy actor-critic method. It optimises a *clipped* surrogate
objective to prevent destructively large policy updates:

    L_CLIP(θ) = 𝔼_t [ min( r_t(θ) Â_t ,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]

    r_t(θ) = π_θ(aₜ | sₜ) / π_{θ_old}(aₜ | sₜ)    (probability ratio)
    Â_t    = GAE-λ advantage estimate

The value function is trained to minimise:
    L_V(φ) = 𝔼_t [ (V_φ(sₜ) − Rₜ)² ]

An entropy bonus −c₂ H[π_θ(·|sₜ)] encourages exploration.

Combined loss:
    L(θ, φ) = −L_CLIP(θ) + c₁ L_V(φ) − c₂ H[π_θ(·|s)]

Generalised Advantage Estimation (GAE-λ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    δₜ  = rₜ + γ V(sₜ₊₁) − V(sₜ)                   (TD residual)
    Âₜ  = Σ_{k=0}^{T-t} (γλ)^k δₜ₊ₖ               (GAE)

λ=1 → Monte Carlo (unbiased, high variance)
λ=0 → 1-step TD (biased, low variance)
λ≈0.95 → good empirical balance

Intuition
~~~~~~~~~
• On-policy: uses only data collected under π_θ_old, discards after each update.
• Clipping: if the policy changes too much (r_t outside [1-ε, 1+ε]), we ignore
  that gradient — so we never take steps that are "too large".
• Multiple epochs: PPO reuses each rollout for K mini-batch epochs, making it
  more sample-efficient than vanilla policy gradient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import MLP, DiscreteActor, StochasticActor, Critic
from utils.buffers import RolloutBuffer


class PPO:
    """Proximal Policy Optimization — supports both discrete and continuous envs."""

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        hidden=(128, 128),
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,          # ε in clipped objective
        n_epochs=10,           # K epochs per rollout
        batch_size=64,
        rollout_steps=2048,    # steps collected per update cycle
        entropy_coef=0.01,     # c₂: entropy bonus
        value_coef=0.5,        # c₁: value loss coefficient
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.continuous = continuous
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Actor π_θ
        if continuous:
            self.actor = StochasticActor(state_dim, action_dim, hidden, squash=False).to(device)
        else:
            self.actor = DiscreteActor(state_dim, action_dim, hidden).to(device)

        # Critic V_φ(s)
        self.critic = Critic(state_dim, action_dim=0, hidden=hidden).to(device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.buffer = RolloutBuffer(state_dim, action_dim if continuous else 1,
                                    rollout_steps, gamma, lam, device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state):
        """Sample action from π_θ(·|s). Returns (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.continuous:
                action, log_prob, _ = self.actor.get_action(state_t)
            else:
                dist = self.actor(state_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            value = self.critic(state_t)

        action_np = action.cpu().numpy().flatten()
        if action_np.shape == (1,) and not self.continuous:
            action_np = action_np[0]
        return action_np, log_prob.cpu().item(), value.cpu().item()

    # ------------------------------------------------------------------
    # Update step (called after each rollout)
    # ------------------------------------------------------------------

    def update(self):
        """PPO update: K epochs of mini-batch gradient descent."""
        states, actions, old_log_probs, advantages, returns = self.buffer.get()

        for _ in range(self.n_epochs):
            # Mini-batch sampling
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                s  = states[idx]
                a  = actions[idx]
                lp = old_log_probs[idx]
                adv = advantages[idx]
                ret = returns[idx]

                # --- Actor loss (clipped surrogate) ---
                if self.continuous:
                    new_action, new_log_prob, _ = self.actor.get_action(s)
                    # re-evaluate log π_θ(aₜ | sₜ) for sampled actions
                    mu, log_std = self.actor(s)
                    std = log_std.exp()
                    from torch.distributions import Normal
                    dist = Normal(mu, std)
                    new_log_prob = dist.log_prob(a).sum(-1, keepdim=True)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    dist = self.actor(s)
                    new_log_prob = dist.log_prob(a.squeeze(-1)).unsqueeze(-1)
                    entropy = dist.entropy().mean()

                ratio = (new_log_prob - lp.unsqueeze(-1)).exp()  # r_t(θ)
                adv_u = adv.unsqueeze(-1)
                surr1 = ratio * adv_u
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_u
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic loss ---
                value_pred = self.critic(s)
                critic_loss = ((value_pred - ret.unsqueeze(-1)) ** 2).mean()

                # --- Total loss ---
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        self.buffer.reset()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_name="CartPole-v1", n_episodes=400, max_steps=500):
        """Full training loop. Returns list of episode rewards."""
        env = gym.make(env_name)
        rewards = []
        ep_reward = 0
        state, _ = env.reset()
        ep_count = 0
        step = 0

        while ep_count < n_episodes:
            action, log_prob, value = self.select_action(state)
            if not self.continuous:
                env_action = int(action)
            else:
                env_action = action
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            self.buffer.add(
                state,
                [action] if not self.continuous else action,
                reward,
                float(done),
                value,
                log_prob,
            )

            state = next_state
            ep_reward += reward
            step += 1

            if done:
                state, _ = env.reset()
                rewards.append(ep_reward)
                ep_reward = 0
                ep_count += 1
                if ep_count % 50 == 0:
                    avg = np.mean(rewards[-50:])
                    print(f"[PPO] Episode {ep_count:4d} | Avg reward (last 50): {avg:.1f}")

            # Update after collecting rollout_steps transitions
            if step % self.rollout_steps == 0:
                with torch.no_grad():
                    last_val = self.critic(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    ).item()
                self.buffer.compute_returns_and_advantages(last_val)
                self.update()

        env.close()
        return rewards
