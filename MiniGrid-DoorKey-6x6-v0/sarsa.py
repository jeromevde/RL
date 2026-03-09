"""
SARSA on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
SARSA = State-Action-Reward-State-Action.  The simplest ON-POLICY TD method.
It keeps a table Q(s, a) and updates it toward the reward + the Q-value of the
action *actually taken* next (not the best one — that's Q-learning).  Because
the update follows the same policy being used to explore, SARSA learns a policy
that accounts for its own exploration noise: it is "cautious" near bad
states it might stumble into.  Family: tabular, model-free, on-policy, TD(0).

Update rule:  Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]
              where a' is the action the ε-greedy policy actually picks.
──────────────────────────────────────────────────────────────────────────

State  : (agent_x, agent_y, agent_dir, has_key, door_open)  — tabular
Actions: full action set (0-6): turn-left, turn-right, forward,
         pickup, drop, toggle, done
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
RESULTS_FILE = os.path.join(RESULTS_DIR, "sarsa_doorkey_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "sarsa_doorkey.png")
WINDOW       = 200
# ────────────────────────────────────────────────────────────────────────────


def get_state(env) -> tuple:
    """Compact symbolic state: (x, y, dir, has_key, door_open)."""
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
        state  = get_state(env)
        action = epsilon_greedy(Q, state, eps)
        total_reward = 0.0
        done = False

        while not done:
            _, reward, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            next_state  = get_state(env)
            next_action = epsilon_greedy(Q, next_state, eps)

            # SARSA on-policy update
            td_target = reward + GAMMA * Q[next_state][next_action] * (not done)
            Q[state][action] += ALPHA * (td_target - Q[state][action])

            state  = next_state
            action = next_action
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
    ax.plot(rewards, alpha=0.2, color="steelblue", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="steelblue",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"SARSA — {ENV_ID}")
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
parser = argparse.ArgumentParser(description=f"SARSA on {ENV_ID}")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("=" * 55)
print(f"  SARSA  |  {ENV_ID}")
print("=" * 55)

rewards, Q = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    render_greedy(Q)
