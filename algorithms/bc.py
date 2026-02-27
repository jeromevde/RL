"""
Behaviour Cloning (BC)
=======================
Reference: Pomerleau, 1989 — "ALVINN: An Autonomous Land Vehicle in a Neural Network"
           Bain & Sammut, 1995 — "A Framework for Behavioural Cloning"

Theory
------
Behaviour Cloning (BC) is the simplest form of imitation learning: treat it as
supervised learning over expert (state, action) pairs.

Given a dataset of expert demonstrations D_E = {(sₜ, aₜ)}, we minimise:

    Discrete actions (cross-entropy):
        L_BC(θ) = −𝔼_{(s,a)~D_E} [ log π_θ(a | s) ]

    Continuous actions (MSE / negative log-likelihood of Gaussian):
        L_BC(θ) = 𝔼_{(s,a)~D_E} [ ‖μ_θ(s) − a‖² ]

Intuition
~~~~~~~~~
• BC is fast and simple — no need to run any RL algorithm.
• It works well when expert demonstrations are plentiful and diverse.
• Key weakness: **distributional shift** — at test time the policy visits states
  never seen during training (because the expert's corrections kept it on track),
  leading to compounding errors (covariate shift).
• DAgger (Ross et al., 2011) fixes this by iteratively querying the expert on
  states visited by the *current* policy.

Workflow
~~~~~~~~
1. Collect expert trajectories (here using a trained SAC/PPO agent).
2. Train π_θ on (s, a) pairs with supervised learning.
3. Evaluate the cloned policy in the environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.networks import MLP, DeterministicActor


class BehaviourCloning:
    """Behaviour Cloning agent for continuous action spaces."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        hidden=(256, 256),
        lr=1e-3,
        device="cpu",
    ):
        self.max_action = max_action
        self.device = device

        # Simple MLP policy: s → a  (deterministic)
        self.policy = DeterministicActor(state_dim, action_dim, max_action, hidden).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Collect expert demonstrations
    # ------------------------------------------------------------------

    def collect_expert_demos(self, expert_agent, env_name, n_episodes=20, max_steps=200):
        """Run `expert_agent` in the environment and record (s, a) pairs.

        Works with any agent that implements `select_action(state)` — the method
        is called with no keyword arguments to be compatible with DQN, DDPG, SAC,
        PPO, and BC agents alike.  If the agent returns a tuple (e.g. PPO returns
        action, log_prob, value), only the first element is used.

        Args:
            expert_agent: any agent with a `select_action(state)` method.
            env_name:     Gymnasium environment name.
            n_episodes:   number of expert episodes to collect.

        Returns:
            states:  np.ndarray (N, state_dim)
            actions: np.ndarray (N, action_dim)
        """
        env = gym.make(env_name)
        all_states, all_actions = [], []

        for _ in range(n_episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                result = expert_agent.select_action(state)
                # Handle agents that return (action, log_prob, value) tuples (e.g. PPO)
                action = result[0] if isinstance(result, tuple) else result
                all_states.append(state)
                all_actions.append(np.atleast_1d(action))
                next_state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                state = next_state

        env.close()
        return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Supervised training on expert data
    # ------------------------------------------------------------------

    def fit(self, states, actions, n_epochs=50, batch_size=64):
        """Train π_θ on expert (s, a) pairs with MSE loss.

        L_BC(θ) = 𝔼[(μ_θ(s) − a_expert)²]
        """
        states_t  = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        N = len(states_t)

        for epoch in range(n_epochs):
            idx = torch.randperm(N)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]
                s, a = states_t[b], actions_t[b]
                pred = self.policy(s)
                loss = ((pred - a) ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"[BC] Epoch {epoch+1:3d}/{n_epochs} | Loss: {epoch_loss/batches:.4f}")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, eval_mode=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state_t).cpu().numpy().flatten()
        return np.clip(action, -self.max_action, self.max_action)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env_name, n_episodes=20, max_steps=200):
        """Evaluate the cloned policy. Returns list of episode rewards."""
        env = gym.make(env_name)
        rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            ep_reward = 0
            for _ in range(max_steps):
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)

        env.close()
        return rewards
