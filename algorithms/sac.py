"""
Soft Actor-Critic (SAC)
========================
Paper: Haarnoja et al., 2018 — "Soft Actor-Critic: Off-Policy Maximum Entropy
       Deep Reinforcement Learning with a Stochastic Actor"
       https://arxiv.org/abs/1801.01290

Paper (auto-α): Haarnoja et al., 2018 — "Soft Actor-Critic Algorithms and Applications"
       https://arxiv.org/abs/1812.05905

Environment: Pendulum-v1  (state ∈ ℝ³,  action ∈ [−2, 2])

Theory
------
SAC augments the standard RL objective with an entropy term:

    J(π) = 𝔼_{τ~π} [ Σ_t  γᵗ ( r(sₜ, aₜ) + α H(π(·|sₜ)) ) ]

    H(π(·|s)) = −𝔼_{a~π} [log π(a|s)]     (entropy)
    α          = temperature (controls exploration vs. exploitation trade-off)

This encourages the policy to be as random as possible while still
maximising reward — a form of exploration regularisation.

**Soft Bellman equation** (critic):
    Q*(s, a) = r + γ 𝔼_{s'} [ V*(s') ]
    V*(s)    = 𝔼_{a~π} [ Q*(s, a) − α log π(a|s) ]

**Critic update** (twin Q-networks to reduce overestimation bias):
    y = r + γ ( min(Q₁', Q₂')(s', ã') − α log π(ã'|s') )
    L_Q = 𝔼 [ (Qᵢ(s, a) − y)²  for i ∈ {1, 2} ]

**Actor update** (reparameterisation trick):
    L_π = 𝔼_{s~D, ε~N} [ α log π(f(ε, s)|s) − min Q(s, f(ε, s)) ]

**Automatic entropy tuning** (target entropy Ĥ = −|A|):
    L_α = 𝔼_{a~π} [ −α ( log π(a|s) + Ĥ ) ]

Intuition
~~~~~~~~~
• SAC learns a *stochastic* policy, unlike DDPG's deterministic one.
• The entropy bonus naturally handles exploration — no need for hand-tuned noise.
• Twin critics prevent the Q-value over-estimation that plagues actor-critic methods.
• Automatic α-tuning keeps entropy at the target level without manual tuning.
• Off-policy → much more sample efficient than PPO.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import copy

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import StochasticActor, TwinCritic
from utils.buffers import ReplayBuffer


class SAC:
    """Soft Actor-Critic agent with automatic entropy tuning."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        hidden=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,             # initial entropy temperature
        auto_entropy=True,     # learn α automatically
        target_entropy=None,   # defaults to −action_dim
        buffer_size=100_000,
        batch_size=256,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device

        # Actor π_θ (stochastic, tanh-squashed Gaussian)
        self.actor = StochasticActor(state_dim, action_dim, hidden, squash=True).to(device)

        # Twin critics Q₁, Q₂ (θ_Q1, θ_Q2) and their targets
        self.critic = TwinCritic(state_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy temperature α
        self.auto_entropy = auto_entropy
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32,
                                      requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.target_entropy = target_entropy or -action_dim
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

    # ------------------------------------------------------------------
    # Soft target update
    # ------------------------------------------------------------------

    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, eval_mode=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(state_t)
        a = (mean if eval_mode else action).cpu().numpy().flatten()
        return np.clip(a * self.max_action, -self.max_action, self.max_action)

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        self.alpha = self.log_alpha.exp().item()

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(next_states)
            next_actions_scaled = next_actions * self.max_action
            target_q = self.critic_target.min_q(next_states, next_actions_scaled)
            # Soft Bellman target includes entropy bonus
            y = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs, _ = self.actor.get_action(states)
        new_actions_scaled = new_actions * self.max_action
        q_min = self.critic.min_q(states, new_actions_scaled)
        actor_loss = (self.alpha * log_probs - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Automatic entropy tuning ---
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update()
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
                print(f"[SAC] Episode {ep+1:4d} | Avg reward (last 20): {avg:.1f} | α={self.alpha:.4f}")

        env.close()
        return rewards
