"""
REINFORCE (Monte-Carlo Policy Gradient) on MiniGrid-Empty-5x5-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
REINFORCE = the simplest POLICY GRADIENT method (Williams, 1992).  Instead of
learning a value table/function and deriving a policy from it, REINFORCE
*directly* parameterises the policy π_θ(a|s) as a neural net and optimises it
by gradient ascent on expected return.  It waits until the end of a full
episode (Monte Carlo), computes the discounted return G_t for each step, and
pushes up the log-probability of actions proportional to how good their G_t
was.  Simple but high-variance — every estimate depends on one full trajectory.
No value function, no bootstrapping, no replay buffer.
Family: neural, model-free, on-policy, Monte-Carlo policy gradient.

Gradient:  ∇θ J ≈ Σ_t ∇θ log π_θ(a_t|s_t) · G_t
           ("make good actions more likely, bad actions less likely")
──────────────────────────────────────────────────────────────────────────

State  : one-hot encoding of (agent_x, agent_y, agent_dir)
Actions: 0=turn-left, 1=turn-right, 2=forward
"""

import argparse
import os

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES   = 3_000
LR           = 5e-3       # learning rate for the policy network
GAMMA        = 0.99
N_ACTIONS    = 3
GRID_W       = 5          # MiniGrid-Empty-5x5 grid width
GRID_H       = 5          # MiniGrid-Empty-5x5 grid height
N_DIRS       = 4
STATE_DIM    = GRID_W * GRID_H * N_DIRS   # = 100 (one-hot size)
ENV_ID       = "MiniGrid-Empty-5x5-v0"
RESULTS_DIR  = "results"
PLOTS_DIR    = "plots"
RESULTS_FILE = os.path.join(RESULTS_DIR, "reinforce_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "reinforce.png")
WINDOW       = 50
# ────────────────────────────────────────────────────────────────────────────


def get_state_tensor(env) -> torch.Tensor:
    """Return a one-hot state vector for the agent's (x, y, direction)."""
    x, y = env.unwrapped.agent_pos
    d    = env.unwrapped.agent_dir
    idx  = int(x) * GRID_H * N_DIRS + int(y) * N_DIRS + int(d)
    one_hot = torch.zeros(STATE_DIM)
    one_hot[idx] = 1.0
    return one_hot


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    """Discounted returns G_t = Σ_{k≥t} γ^(k-t) r_k, then standardised."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Standardise to reduce variance
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train():
    env    = gym.make(ENV_ID)
    policy = PolicyNet()
    opt    = optim.Adam(policy.parameters(), lr=LR)
    episode_rewards = []

    for ep in range(N_EPISODES):
        env.reset()
        log_probs, rewards = [], []
        done = False

        while not done:
            state  = get_state_tensor(env)
            logits = policy(state)
            dist   = Categorical(logits=logits)
            action = dist.sample()

            _, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Policy gradient update
        returns = compute_returns(rewards, GAMMA)
        loss    = -torch.stack(log_probs) @ returns   # scalar
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:4d}/{N_EPISODES}  loss={loss.item():.4f}  avg-reward(last 100)={avg:.3f}")

    env.close()
    return episode_rewards, policy


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, color="seagreen", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="seagreen",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"REINFORCE — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


def render_greedy(policy: PolicyNet):
    env = gym.make(ENV_ID, render_mode="human")
    env.reset()
    done  = False
    total = 0.0
    with torch.no_grad():
        while not done:
            state  = get_state_tensor(env)
            logits = policy(state)
            action = logits.argmax().item()
            _, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += reward
    env.close()
    print(f"  Greedy episode reward: {total:.3f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="REINFORCE on MiniGrid-Empty-5x5-v0")
parser.add_argument("--render", action="store_true",
                    help="Render one greedy episode after training")
args = parser.parse_args()

print("=" * 50)
print("  REINFORCE  |  " + ENV_ID)
print("=" * 50)

rewards, policy = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    print("\n  Rendering greedy policy …")
    render_greedy(policy)
