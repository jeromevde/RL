"""
Shared neural network building blocks.

All algorithms use simple MLP architectures — the key insight is that with
modern deep learning, even 2-layer networks can represent rich policies and
value functions for low-dimensional environments like CartPole and Pendulum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


def MLP(input_dim, output_dim, hidden=(256, 256), activation=nn.ReLU):
    """Constructs a multi-layer perceptron.

    Architecture: input → [hidden layers] → output (no final activation).
    """
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-network for DQN: maps state → Q-value for every discrete action.

    Q(s, a; θ)  for all a simultaneously.
    """

    def __init__(self, state_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden)

    def forward(self, state):
        return self.net(state)


class DeterministicActor(nn.Module):
    """Deterministic policy μ(s; θ) → a  (used by DDPG).

    Output is squashed to [-max_action, max_action] via tanh.
    """

    def __init__(self, state_dim, action_dim, max_action, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))


class StochasticActor(nn.Module):
    """Gaussian stochastic policy π(a|s; θ)  (used by SAC and PPO continuous).

    Outputs mean μ(s) and log std log σ(s) of a diagonal Gaussian.

    Log-likelihood (for policy gradient):
        log π(a|s) = -½ Σ_i [(aᵢ - μᵢ)² / σᵢ² + log σᵢ² + log 2π]

    For SAC we additionally apply a tanh squashing:
        ã = tanh(u),   u ~ N(μ, σ²)
    with log-prob correction:
        log π(ã|s) = log π(u|s) - Σ_i log(1 - tanh²(uᵢ))
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden=(256, 256), squash=False):
        super().__init__()
        self.squash = squash
        self.net = MLP(state_dim, action_dim * 2, hidden)  # outputs [μ, log σ]

    def forward(self, state):
        out = self.net(state)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def get_action(self, state):
        """Sample action and return (action, log_prob, mean)."""
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = dist.rsample()

        if self.squash:
            action = torch.tanh(u)
            # Jacobian correction for tanh squashing
            log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            action = u
            log_prob = dist.log_prob(u).sum(-1, keepdim=True)

        return action, log_prob, torch.tanh(mu)


class DiscreteActor(nn.Module):
    """Categorical (softmax) policy π(a|s; θ)  (used by PPO discrete).

    log π(a|s) = log softmax(logits(s))[a]
    """

    def __init__(self, state_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.net = MLP(state_dim, n_actions, hidden)

    def forward(self, state):
        return Categorical(logits=self.net(state))


class Critic(nn.Module):
    """State-action value function Q(s, a; θ) or state value V(s; θ).

    For actor-critic methods:
        - V(s): maps state → scalar value estimate
        - Q(s,a): maps (state, action) → scalar Q-value
    """

    def __init__(self, state_dim, action_dim=0, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state, action=None):
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.net(x)


class TwinCritic(nn.Module):
    """Two independent Q-networks Q₁ and Q₂ (used by SAC/TD3 to reduce overestimation).

    TD3/SAC take min(Q₁, Q₂) as target to combat the maximisation bias.
    """

    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden)
        self.q2 = Critic(state_dim, action_dim, hidden)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def min_q(self, state, action):
        q1, q2 = self(state, action)
        return torch.min(q1, q2)
