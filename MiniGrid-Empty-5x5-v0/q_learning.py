"""
Q-Learning on MiniGrid-Empty-5x5-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
Q-Learning = the OFF-POLICY sibling of SARSA.  Both maintain a Q-table, but
Q-learning always updates toward the *best possible* next action (max_a Q),
regardless of what the agent actually did.  This decouples the exploration
policy (ε-greedy) from the learned value (greedy), which is the definition
of off-policy learning.  Converges to the optimal Q* even with random
exploration, given enough visits.  Family: tabular, model-free, off-policy,
TD(0).  DQN is Q-learning with a neural net replacing the table.

Update rule:  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
              (contrast SARSA: uses Q(s',a') where a' is actually taken)
──────────────────────────────────────────────────────────────────────────

State  : (agent_x, agent_y, agent_dir)  — tabular
Actions: 0=turn-left, 1=turn-right, 2=forward  (only these matter in Empty)
"""
#%%
import argparse
import os
from collections import defaultdict

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import matplotlib.pyplot as plt
import numpy as np

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_EPISODES   = 1_000
ALPHA        = 0.1
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.01
EPS_DECAY    = 0.999
N_ACTIONS    = 3
ENV_ID       = "MiniGrid-Empty-5x5-v0"
RESULTS_DIR  = "results"
PLOTS_DIR    = "plots"
RESULTS_FILE = os.path.join(RESULTS_DIR, "q_learning_rewards.npy")
PLOT_FILE    = os.path.join(PLOTS_DIR,   "q_learning.png")
WINDOW       = 50
# ────────────────────────────────────────────────────────────────────────────


def get_state(env) -> tuple:
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
        state        = get_state(env)
        total_reward = 0.0
        done         = False

        while not done:
            action = epsilon_greedy(Q, state, eps)
            _, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = get_state(env)

            # Q-Learning update: target uses greedy max over next state (off-policy)
            td_target = reward + GAMMA * np.max(Q[next_state]) * (not done)
            td_error  = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error

            state        = next_state
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
    ax.plot(rewards, alpha=0.3, color="darkorange", label="episode reward")
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
    print(f"  Greedy episode reward: {total:.3f}")

#%%
# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Q-Learning on MiniGrid-Empty-5x5-v0")
parser.add_argument("--render", action="store_true",
                    help="Render one greedy episode after training")
args = parser.parse_args()

print("=" * 50)
print("  Q-Learning  |  " + ENV_ID)
print("=" * 50)

rewards, Q = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    print("\n  Rendering greedy policy …")
    render_greedy(Q)

#%%
if True:  # debug – run 10 full greedy episodes, print every step
    ACTION_NAMES = {0: "turn-left", 1: "turn-right", 2: "forward"}
    N_DEBUG_EPS  = 10
    N_EPISODES   = 10


    env = gym.make(ENV_ID, render_mode="human")
    rewards, Q = train()

    for ep in range(1, N_DEBUG_EPS + 1):
        env.reset()
        state      = get_state(env)
        done       = False
        step_num   = 0
        ep_reward  = 0.0

        print(f"\n{'='*52}")
        print(f"  Episode {ep}/{N_DEBUG_EPS}")
        print(f"{'='*52}")
        print(f"  Initial state : {state}")

        while not done:
            action = int(np.argmax(Q[state]))
            _, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = get_state(env)
            ep_reward += reward
            step_num  += 1

            print(f"  Step {step_num:3d} | state={state}  action={ACTION_NAMES[action]:11s}"
                  f"  reward={reward:+.3f}  next={next_state}  {'DONE' if done else ''}")

            state = next_state

        print(f"  ── Episode {ep} finished in {step_num} steps | total reward = {ep_reward:+.3f}")

    env.close()

# %%
