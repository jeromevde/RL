"""
REINFORCE (Monte-Carlo Policy Gradient) on MiniGrid-DoorKey-6x6-v0
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

Observation: flattened agent partial-view image (7×7×3 = 147 values), normalised.
             Direction (0-3) is appended → state_dim = 148.
Actions    : 0-6 (full MiniGrid action set).
"""

import argparse
import os

import gymnasium as gym
import minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES   = 10_000
LR           = 3e-4
GAMMA        = 0.99
ENTROPY_COEF = 0.01       # encourages exploration
N_ACTIONS    = 7
ENV_ID       = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR  = "results"
PLOTS_DIR    = "plots"
RESULTS_FILE = os.path.join(RESULTS_DIR, "reinforce_doorkey_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "reinforce_doorkey.png")
WINDOW       = 200
# ────────────────────────────────────────────────────────────────────────────


def obs_to_tensor(obs: dict) -> torch.Tensor:
    img = obs["image"].flatten().astype(np.float32) / 10.0
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([img, direction]))


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train():
    env      = gym.make(ENV_ID)
    obs0, _  = env.reset()
    state_dim = len(obs_to_tensor(obs0))

    policy = PolicyNet(state_dim)
    opt    = optim.Adam(policy.parameters(), lr=LR)
    episode_rewards = []

    for ep in range(N_EPISODES):
        obs, _     = env.reset()
        log_probs  = []
        entropies  = []
        rewards    = []
        done       = False

        while not done:
            state  = obs_to_tensor(obs)
            logits = policy(state)
            dist   = Categorical(logits=logits)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(reward)

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        returns = compute_returns(rewards, GAMMA)
        policy_loss  = -(torch.stack(log_probs) * returns).sum()
        entropy_loss = -ENTROPY_COEF * torch.stack(entropies).sum()
        loss = policy_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

        if (ep + 1) % 1000 == 0:
            avg = np.mean(episode_rewards[-200:])
            print(f"  Episode {ep+1:6d}/{N_EPISODES}  loss={loss.item():.4f}  avg200={avg:.4f}")

    env.close()
    return episode_rewards, policy


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="seagreen", label="episode reward")
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
    obs, _ = env.reset()
    done   = False
    total  = 0.0
    with torch.no_grad():
        while not done:
            state  = obs_to_tensor(obs)
            logits = policy(state)
            action = logits.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += reward
    env.close()
    print(f"  Greedy episode reward: {total:.4f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=f"REINFORCE on {ENV_ID}")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("=" * 55)
print(f"  REINFORCE  |  {ENV_ID}")
print("=" * 55)

rewards, policy = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    render_greedy(policy)
