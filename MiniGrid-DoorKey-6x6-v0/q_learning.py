"""
Q-Learning on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
Q-Learning = the OFF-POLICY sibling of SARSA.  Both maintain a Q-table, but
Q-learning always updates toward the *best possible* next action (max_a Q),
regardless of what the agent actually did.  This decouples the exploration
policy (ε-greedy) from the learned value (greedy), which is the definition
of off-policy learning.  Converges to the optimal Q* even with random
exploration, given enough visits.  Family: tabular, model-free, off-policy,
TD(0).  DQN is this algorithm with a neural net replacing the table.

Update rule:  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
              (contrast SARSA: uses Q(s',a') where a' is actually taken)
──────────────────────────────────────────────────────────────────────────

State  : (agent_x, agent_y, agent_dir, has_key, door_open)  — tabular
Actions: 0-6 (turn-left, turn-right, forward, pickup, drop, toggle, done)
"""

import argparse
import os
from collections import defaultdict

import gymnasium as gym
import minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES   = 20_000
ALPHA        = 0.2
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.02
EPS_DECAY    = 0.9995
N_ACTIONS    = 7
ENV_ID       = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR  = "results"
PLOTS_DIR    = "plots"
RESULTS_FILE = os.path.join(RESULTS_DIR, "q_learning_doorkey_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "q_learning_doorkey.png")
WINDOW       = 200
# ────────────────────────────────────────────────────────────────────────────


def get_state(env) -> tuple:
    x, y     = env.unwrapped.agent_pos
    d        = env.unwrapped.agent_dir
    has_key  = int(env.unwrapped.carrying is not None)
    door_open = 0
    for obj in env.unwrapped.grid.grid:
        if obj is not None and obj.type == "door":
            door_open = int(obj.is_open)
            break
    return (int(x), int(y), int(d), has_key, door_open)


def epsilon_greedy(Q, state, eps: float) -> int:
    if np.random.random() < eps:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[state]))


def train():
    env = gym.make(ENV_ID)
    Q   = defaultdict(lambda: np.zeros(N_ACTIONS))
    eps = EPS_START
    episode_rewards = []

    for ep in range(N_EPISODES):
        env.reset()
        state        = get_state(env)
        total_reward = 0.0
        done         = False

        while not done:
            action = epsilon_greedy(Q, state, eps)
            _, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = get_state(env)

            # Q-Learning off-policy update: max over next state
            td_target = reward + GAMMA * np.max(Q[next_state]) * (not done)
            Q[state][action] += ALPHA * (td_target - Q[state][action])

            state        = next_state
            total_reward += reward

        eps = max(EPS_END, eps * EPS_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % 2000 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"  Episode {ep+1:6d}/{N_EPISODES}  ε={eps:.4f}  avg500={avg:.4f}")

    env.close()
    return episode_rewards, Q


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="darkorange", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="darkorange",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Q-Learning — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


def render_greedy(Q):
    env = gym.make(ENV_ID, render_mode="human")
    env.reset()
    state = get_state(env)
    done  = False
    total = 0.0
    while not done:
        action = int(np.argmax(Q[state]))
        _, reward, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        state = get_state(env)
        total += reward
    env.close()
    print(f"  Greedy episode reward: {total:.4f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=f"Q-Learning on {ENV_ID}")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("=" * 55)
print(f"  Q-Learning  |  {ENV_ID}")
print("=" * 55)

rewards, Q = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    render_greedy(Q)
