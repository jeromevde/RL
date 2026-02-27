"""
Deep Deterministic Policy Gradient (DDPG)
==========================================
Paper: Lillicrap et al., 2015 — "Continuous control with deep reinforcement learning"
       https://arxiv.org/abs/1509.02971

Environment: Pendulum-v1  (state ∈ ℝ³,  action ∈ [−2, 2])

Theory
------
DDPG is an off-policy actor-critic algorithm for *continuous* action spaces.
It extends DQN to continuous actions by maintaining a deterministic policy
μ(s; θ_μ) (the "actor") alongside a Q-value critic Q(s, a; θ_Q).

**Critic update** — minimise Bellman error:
    L(θ_Q) = 𝔼 [ (y − Q(s, a; θ_Q))² ]
    y = r + γ Q(s', μ(s'; θ_μ⁻); θ_Q⁻)       ← target networks

**Actor update** — maximise Q via chain rule (deterministic policy gradient):
    ∇_θ_μ J ≈ 𝔼 [ ∇_a Q(s, a; θ_Q)|_{a=μ(s)} · ∇_θ_μ μ(s; θ_μ) ]

Both actor and critic use *soft* (Polyak) target network updates:
    θ_target ← τ θ + (1 − τ) θ_target          (τ ≪ 1, e.g. 0.005)

Exploration in continuous space is achieved by adding Ornstein-Uhlenbeck noise
to the deterministic action:
    aₜ = μ(sₜ; θ_μ) + 𝒩ₜ

OU noise is temporally correlated (unlike Gaussian noise) which works better
for physical systems with inertia, but simple Gaussian noise often works too.

Intuition
~~~~~~~~~
• "Actor" = the policy (what action to take).
• "Critic" = the judge (how good is that action?).
• The actor is directly improved by the critic's gradient — no stochastic
  sampling needed, which makes gradient variance low.
• The downside: exploration is tricky since the policy is deterministic.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import copy

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import DeterministicActor, Critic
from utils.buffers import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration noise.

    dxₜ = θ(μ − xₜ)dt + σ dWₜ       (mean-reverting Brownian motion)
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state


class DDPG:
    """Deep Deterministic Policy Gradient agent."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden=(256, 256),
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,             # Polyak averaging coefficient
        buffer_size=100_000,
        batch_size=128,
        noise_sigma=0.1,       # Gaussian exploration noise std
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device

        # Actor μ(s; θ_μ) and its target
        self.actor = DeterministicActor(state_dim, action_dim, max_action, hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic Q(s, a; θ_Q) and its target
        self.critic = Critic(state_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)
        self.noise = OUNoise(action_dim)
        self.noise_sigma = noise_sigma

    # ------------------------------------------------------------------
    # Soft target update: θ_target ← τ θ + (1−τ) θ_target
    # ------------------------------------------------------------------

    def _soft_update(self, net, target):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, add_noise=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        if add_noise:
            action = action + self.noise_sigma * np.random.randn(*action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update (DPG theorem) ---
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft updates of target networks ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_name="Pendulum-v1", n_episodes=200, max_steps=200, warmup_steps=1000):
        """Full training loop. Returns list of episode rewards."""
        env = gym.make(env_name)
        rewards = []
        total_steps = 0

        for ep in range(n_episodes):
            state, _ = env.reset()
            self.noise.reset()
            ep_reward = 0

            for _ in range(max_steps):
                if total_steps < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.add(state, action, reward, next_state, float(done))
                self.update()

                state = next_state
                ep_reward += reward
                total_steps += 1
                if done:
                    break

            rewards.append(ep_reward)
            if (ep + 1) % 20 == 0:
                avg = np.mean(rewards[-20:])
                print(f"[DDPG] Episode {ep+1:4d} | Avg reward (last 20): {avg:.1f}")

        env.close()
        return rewards
