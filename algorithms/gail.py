"""
Generative Adversarial Imitation Learning (GAIL)
=================================================
Paper: Ho & Ermon, 2016 — "Generative Adversarial Imitation Learning"
       https://arxiv.org/abs/1606.03476

Environment: Pendulum-v1

Theory
------
GAIL frames imitation learning as a game between two networks:

    Discriminator D_w(s, a): classifies (s,a) as "expert" (1) or "policy" (0).
    Generator (policy) π_θ:  tries to fool the discriminator.

The discriminator is trained with binary cross-entropy:

    L_D = −𝔼_{(s,a)~π_E}[log D_w(s,a)] − 𝔼_{(s,a)~π_θ}[log(1 − D_w(s,a))]

The policy is updated with RL using the *discriminator reward*:

    r̃(s, a) = −log(1 − D_w(s, a))     (reward increases as D is fooled)

or equivalently:
    r̃(s, a) = log D_w(s, a)            (log-probability of being "expert")

The policy update uses PPO with discriminator rewards, replacing the true
environment reward r(s, a).

Connection to IRL
~~~~~~~~~~~~~~~~~
GAIL implicitly learns a reward function via the discriminator, similar to
Inverse Reinforcement Learning (IRL). But unlike IRL:
• No explicit reward function is extracted.
• The whole loop is end-to-end differentiable.
• Much more scalable than traditional IRL methods.

Intuition
~~~~~~~~~
• Like GAN for images: the "generator" (policy) learns to produce
  (state, action) pairs indistinguishable from the expert.
• Key advantage over BC: avoids distributional shift — the policy is trained
  with RL, so it visits and learns from its *own* states.
• Weakness: harder to train (GAN instability), needs many env interactions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import MLP
from utils.buffers import RolloutBuffer
from algorithms.ppo import PPO


class Discriminator(nn.Module):
    """Binary classifier D_w(s, a) → probability of being expert.

    Input: (state, action) concatenated.
    Output: scalar ∈ (0, 1).
    """

    def __init__(self, state_dim, action_dim, hidden=(128, 128)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return torch.sigmoid(self.net(x))


class GAIL:
    """Generative Adversarial Imitation Learning.

    Uses PPO as the RL backbone. Expert demonstrations must be provided
    as (states, actions) arrays (e.g. from BehaviourCloning.collect_expert_demos).
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        expert_states,
        expert_actions,
        max_action=1.0,
        hidden=(128, 128),
        lr_disc=1e-4,
        disc_update_steps=5,   # discriminator updates per PPO update
        device="cpu",
        ppo_kwargs=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.disc_update_steps = disc_update_steps

        # Store expert data as tensors
        self.expert_states  = torch.FloatTensor(expert_states).to(device)
        self.expert_actions = torch.FloatTensor(expert_actions).to(device)

        # Discriminator D_w
        self.disc = Discriminator(state_dim, action_dim, hidden).to(device)
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=lr_disc)

        # PPO agent (generator)
        ppo_kwargs = ppo_kwargs or {}
        self.ppo = PPO(
            state_dim, action_dim, continuous=True,
            hidden=hidden, device=device, **ppo_kwargs
        )

    # ------------------------------------------------------------------
    # Discriminator training
    # ------------------------------------------------------------------

    def _update_discriminator(self, policy_states, policy_actions):
        """Train D_w to distinguish expert from policy trajectories.

        L_D = −E_expert[log D] − E_policy[log(1−D)]
        """
        batch_size = min(len(policy_states), len(self.expert_states))
        expert_idx = torch.randint(0, len(self.expert_states), (batch_size,))
        expert_s = self.expert_states[expert_idx]
        expert_a = self.expert_actions[expert_idx]

        policy_idx = torch.randint(0, len(policy_states), (batch_size,))
        policy_s = policy_states[policy_idx].to(self.device)
        policy_a = policy_actions[policy_idx].to(self.device)

        expert_prob  = self.disc(expert_s, expert_a)
        policy_prob  = self.disc(policy_s, policy_a)

        ones  = torch.ones_like(expert_prob)
        zeros = torch.zeros_like(policy_prob)

        loss = F.binary_cross_entropy(expert_prob, ones) + \
               F.binary_cross_entropy(policy_prob, zeros)

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Discriminator reward
    # ------------------------------------------------------------------

    def _disc_reward(self, state, action):
        """Surrogate reward r̃(s,a) = log D_w(s,a) (higher → more expert-like)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.disc(s, a)
        return torch.log(prob + 1e-8).item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_name="Pendulum-v1", n_episodes=200, max_steps=200):
        """Full GAIL training loop. Returns list of episode discriminator-rewards."""
        env = gym.make(env_name)
        disc_rewards = []
        ep_count = 0
        step = 0
        state, _ = env.reset()
        ep_reward = 0.0
        rollout_states, rollout_actions = [], []

        while ep_count < n_episodes:
            action, log_prob, value = self.ppo.select_action(state)
            env_action = np.clip(action * self.max_action, -self.max_action, self.max_action)

            next_state, _, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            # Replace environment reward with discriminator reward
            reward = self._disc_reward(state, env_action)
            ep_reward += reward

            self.ppo.buffer.add(
                state,
                action,
                reward,
                float(done),
                value,
                log_prob,
            )
            rollout_states.append(state)
            rollout_actions.append(env_action)

            state = next_state
            step += 1

            if done:
                state, _ = env.reset()
                disc_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_count += 1
                if ep_count % 20 == 0:
                    avg = np.mean(disc_rewards[-20:])
                    print(f"[GAIL] Episode {ep_count:4d} | Avg disc reward (last 20): {avg:.3f}")

            # PPO + Discriminator update
            if step % self.ppo.rollout_steps == 0:
                with torch.no_grad():
                    last_val = self.ppo.critic(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    ).item()
                self.ppo.buffer.compute_returns_and_advantages(last_val)
                self.ppo.update()

                ps = torch.FloatTensor(np.array(rollout_states))
                pa = torch.FloatTensor(np.array(rollout_actions))
                for _ in range(self.disc_update_steps):
                    self._update_discriminator(ps, pa)
                rollout_states, rollout_actions = [], []

        env.close()
        return disc_rewards

    def select_action(self, state, eval_mode=False):
        """Delegate to the inner PPO actor."""
        action, _, _ = self.ppo.select_action(state)
        return np.clip(action * self.max_action, -self.max_action, self.max_action)
