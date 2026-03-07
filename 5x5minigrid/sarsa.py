"""
SARSA on MiniGrid-Empty-5x5-v0
Standalone script — no shared dependencies.

State  : (agent_x, agent_y, agent_dir)  — tabular
Actions: 0=turn-left, 1=turn-right, 2=forward  (only these matter in Empty)
"""

import argparse
import os
from collections import defaultdict

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import matplotlib.pyplot as plt
import numpy as np

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES   = 3_000
ALPHA        = 0.1        # learning rate
GAMMA        = 0.99       # discount factor
EPS_START    = 1.0        # initial ε
EPS_END      = 0.01       # final ε
EPS_DECAY    = 0.999      # multiplicative decay per episode
N_ACTIONS    = 3          # only left / right / forward
ENV_ID       = "MiniGrid-Empty-5x5-v0"
RESULTS_DIR  = "results"
PLOTS_DIR    = "plots"
RESULTS_FILE = os.path.join(RESULTS_DIR, "sarsa_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "sarsa.png")
WINDOW       = 50         # smoothing window for the plot
# ────────────────────────────────────────────────────────────────────────────


def get_state(env) -> tuple:
    """Extract (x, y, direction) from the unwrapped environment."""
    x, y = env.unwrapped.agent_pos
    d    = env.unwrapped.agent_dir
    return (int(x), int(y), int(d))


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
            obs, reward, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            next_state  = get_state(env)
            next_action = epsilon_greedy(Q, next_state, eps)

            # SARSA update: uses the *chosen* next action (on-policy)
            td_target = reward + GAMMA * Q[next_state][next_action] * (not done)
            td_error  = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error

            state   = next_state
            action  = next_action
            total_reward += reward

        eps = max(EPS_END, eps * EPS_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:4d}/{N_EPISODES}  ε={eps:.3f}  avg-reward(last 100)={avg:.3f}")

    env.close()
    return episode_rewards, Q


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="episode reward")
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
    """Run one greedy episode with the learned Q-table and render it."""
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
    print(f"  Greedy episode reward: {total:.3f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SARSA on MiniGrid-Empty-5x5-v0")
parser.add_argument("--render", action="store_true",
                    help="Render one greedy episode after training")
args = parser.parse_args()

print("=" * 50)
print("  SARSA  |  " + ENV_ID)
print("=" * 50)

rewards, Q = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    print("\n  Rendering greedy policy …")
    render_greedy(Q)
