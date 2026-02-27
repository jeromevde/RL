"""
Experience replay buffers.

Off-policy algorithms (DQN, DDPG, SAC) store all past transitions in a
ReplayBuffer and sample random mini-batches — this breaks temporal correlations
and dramatically improves sample efficiency.

On-policy algorithms (PPO) collect a fixed-length RolloutBuffer of the current
policy's trajectories, compute advantages, update, then discard the buffer.
"""

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer for (s, a, r, s', done) transitions.

    Used by off-policy algorithms: DQN, DDPG, SAC, and BC.

    Sampling uniformly at random breaks the temporal correlation that would
    otherwise cause training instability (Mnih et al., 2015, §4).
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


class RolloutBuffer:
    """Fixed-size buffer for on-policy rollouts.

    Stores (s, a, r, done, value, log_prob) from the *current* policy.
    After a full rollout, computes returns and GAE advantages, then is cleared.

    Generalised Advantage Estimation (Schulman et al., 2015b):
        δₜ = rₜ + γ V(sₜ₊₁) - V(sₜ)         (TD residual)
        Âₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + …  (GAE)

    λ=1 → Monte Carlo returns (high variance, low bias)
    λ=0 → one-step TD (low variance, high bias)
    """

    def __init__(self, state_dim, action_dim, max_size, gamma=0.99, lam=0.95, device="cpu"):
        self.max_size = max_size
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.ptr = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.advantages = np.zeros(max_size, dtype=np.float32)
        self.returns = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, done, value, log_prob):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value):
        """Compute GAE(λ) advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get(self):
        n = self.ptr
        states = torch.FloatTensor(self.states[:n]).to(self.device)
        actions = torch.FloatTensor(self.actions[:n]).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs[:n]).to(self.device)
        advantages = torch.FloatTensor(self.advantages[:n]).to(self.device)
        returns = torch.FloatTensor(self.returns[:n]).to(self.device)
        # Normalise advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, log_probs, advantages, returns

    def reset(self):
        self.ptr = 0
