"""
DQN (Deep Q-Network) on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
DQN = Q-Learning with a neural network instead of a table (Mnih et al., 2015).
Tabular Q-learning stores one number per (state, action) pair — impossible when
states are images or high-dimensional vectors.  DQN approximates Q(s,a) with a
neural net and adds two tricks to make it stable:
  1. REPLAY BUFFER — store transitions and sample random mini-batches,
     breaking correlation between consecutive samples (off-policy).
  2. TARGET NETWORK — a frozen copy of the Q-net used to compute TD targets,
     updated periodically, preventing the "chasing a moving target" problem.
This script also uses the Double DQN trick (Hasselt 2016): the online net
picks the best action, but the target net evaluates it, reducing Q-value
overestimation.
Family: neural, model-free, off-policy, value-based, TD.

Relation to other algorithms:
  Q-Learning → (add neural net + replay + target net) → DQN
──────────────────────────────────────────────────────────────────────────

Observation: flattened partial-view image (7×7×3 = 147) + direction → 148 dims.
Actions    : 0-6 (full MiniGrid action set).
"""

import argparse
import os
import random
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES         = 5_000
LR                 = 1e-3
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY          = 0.9995
BUFFER_SIZE        = 50_000
BATCH_SIZE         = 64
TARGET_UPDATE_FREQ = 500        # steps between target net hard updates
LEARN_START        = 1_000      # steps before first gradient update
N_ACTIONS          = 7
ENV_ID             = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR        = "results"
PLOTS_DIR          = "plots"
RESULTS_FILE       = os.path.join(RESULTS_DIR, "dqn_doorkey_rewards.npy")
PLOT_FILE          = os.path.join(PLOTS_DIR,   "dqn_doorkey.png")
WINDOW             = 100
# ────────────────────────────────────────────────────────────────────────────


def obs_to_np(obs: dict) -> np.ndarray:
    img = obs["image"].flatten().astype(np.float32) / 10.0
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
    return np.concatenate([img, direction])


class QNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def train():
    env      = gym.make(ENV_ID)
    obs0, _  = env.reset()
    state_dim = len(obs_to_np(obs0))

    online_net = QNet(state_dim)
    target_net = QNet(state_dim)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    opt    = optim.Adam(online_net.parameters(), lr=LR)
    buf    = ReplayBuffer(BUFFER_SIZE)
    eps    = EPS_START
    step   = 0
    episode_rewards = []

    for ep in range(N_EPISODES):
        obs, _       = env.reset()
        state        = obs_to_np(obs)
        total_reward = 0.0
        done         = False

        while not done:
            # ε-greedy action
            if np.random.random() < eps or step < LEARN_START:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = online_net(torch.from_numpy(state).unsqueeze(0))
                action = q.argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = obs_to_np(obs)

            buf.push(state, action, reward, next_state, float(done))
            state        = next_state
            total_reward += reward
            step         += 1

            # Learning step
            if step >= LEARN_START and len(buf) >= BATCH_SIZE:
                s, a, r, s2, d = buf.sample(BATCH_SIZE)

                with torch.no_grad():
                    # Double DQN: online selects action, target evaluates it
                    best_actions = online_net(s2).argmax(dim=1, keepdim=True)
                    # IPython.embed(header="Double DQN target computation debug breakpoint")
                    q_next       = target_net(s2).gather(1, best_actions).squeeze(1)
                    q_target     = r + GAMMA * q_next * (1.0 - d)

                q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, q_target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                opt.step()

            if step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

        eps = max(EPS_END, eps * EPS_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:5d}/{N_EPISODES}  step={step:7d}  ε={eps:.3f}  avg100={avg:.4f}")

    env.close()
    return episode_rewards, online_net


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="crimson", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="crimson",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"DQN — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


def render_greedy(net: QNet):
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    done   = False
    total  = 0.0
    with torch.no_grad():
        while not done:
            state  = torch.from_numpy(obs_to_np(obs)).unsqueeze(0)
            action = net(state).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += reward
    env.close()
    print(f"  Greedy episode reward: {total:.4f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=f"DQN on {ENV_ID}")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("=" * 55)
print(f"  DQN  |  {ENV_ID}")
print("=" * 55)

rewards, net = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    render_greedy(net)
