"""
Deep Q-Network (DQN)
====================
Paper: Mnih et al., 2015 — "Human-level control through deep reinforcement learning"
       https://www.nature.com/articles/nature14236

Environment: CartPole-v1  (state ∈ ℝ⁴,  actions ∈ {LEFT, RIGHT})

Theory
------
Q-learning maintains an action-value function Q(s, a) satisfying the
Bellman optimality equation:

    Q*(s, a) = 𝔼[r + γ max_{a'} Q*(s', a')  |  s, a]

DQN approximates Q* with a neural network Q(s, a; θ) and minimises:

    L(θ) = 𝔼_{(s,a,r,s') ~ D} [(y - Q(s, a; θ))²]

    y = r + γ max_{a'} Q(s', a'; θ⁻)           ← Bellman target

where θ⁻ are the *target network* parameters, copied from θ every C steps.

Key ideas
~~~~~~~~~
1. **Experience replay** (buffer D): store all past (s, a, r, s') transitions
   and sample mini-batches uniformly → breaks temporal correlations.

2. **Target network** (θ⁻): a separate, periodically-updated copy of θ used
   only to compute targets. Prevents chasing a moving target.

3. **ε-greedy exploration**: with probability ε take a random action; otherwise
   act greedily: a = argmax_a Q(s, a; θ). ε decays over time.

Notation match with the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
θ   → self.q_net parameters
θ⁻  → self.target_net parameters  (synced every `target_update_freq` steps)
D   → self.buffer (ReplayBuffer)
ε   → self.epsilon  (annealed from epsilon_start → epsilon_end)
γ   → self.gamma
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import QNetwork
from utils.buffers import ReplayBuffer


class DQN:
    """Deep Q-Network agent for discrete action spaces."""

    def __init__(
        self,
        state_dim,
        n_actions,
        hidden=(128, 128),
        lr=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=500,   # steps between θ⁻ ← θ hard updates
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device
        self.steps = 0

        # Q-network θ and target network θ⁻
        self.q_net = QNetwork(state_dim, n_actions, hidden).to(device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(state_dim, 1, buffer_size, device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, eval_mode=False):
        """ε-greedy action selection.

        During evaluation (eval_mode=True) always act greedily.
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self):
        """One gradient update step on a mini-batch from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        actions = actions.long()

        # Current Q-values: Q(s, a; θ)
        q_values = self.q_net(states).gather(1, actions)

        # Bellman target: y = r + γ max_{a'} Q(s', a'; θ⁻)
        with torch.no_grad():
            max_q_next = self.target_net(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * max_q_next * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (common DQN trick)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Hard update of target network every C steps
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay ε
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env_name="CartPole-v1", n_episodes=400, max_steps=500):
        """Full training loop. Returns list of episode rewards."""
        env = gym.make(env_name)
        rewards = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_reward = 0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.add(state, [action], reward, next_state, float(done))
                self.update()

                state = next_state
                ep_reward += reward
                if done:
                    break

            rewards.append(ep_reward)
            if (ep + 1) % 50 == 0:
                avg = np.mean(rewards[-50:])
                print(f"[DQN] Episode {ep+1:4d} | Avg reward (last 50): {avg:.1f} | ε={self.epsilon:.3f}")

        env.close()
        return rewards
