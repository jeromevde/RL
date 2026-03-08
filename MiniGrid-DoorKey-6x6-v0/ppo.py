"""
PPO (Proximal Policy Optimization) on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

Observation: flattened partial-view image (7×7×3 = 147) + direction → 148 dims.
Actions    : 0-6 (full MiniGrid action set).

Algorithm:
  1. Collect a rollout of ROLLOUT_STEPS steps.
  2. Compute GAE advantages + normalise.
  3. Update for PPO_EPOCHS epochs with mini-batches.
  4. Clipped surrogate objective + entropy bonus + value loss.
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
N_EPISODES     = 5_000
ROLLOUT_STEPS  = 128          # steps collected per update cycle
PPO_EPOCHS     = 4            # optimisation epochs per rollout
MINI_BATCH     = 32
LR             = 2.5e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95         # λ for Generalised Advantage Estimation
CLIP_EPS       = 0.2          # PPO clip parameter
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
MAX_GRAD_NORM  = 0.5
N_ACTIONS      = 7
ENV_ID         = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR    = "results"
PLOTS_DIR      = "plots"
RESULTS_FILE   = os.path.join(RESULTS_DIR, "ppo_doorkey_rewards.npy")
PLOT_FILE      = os.path.join(PLOTS_DIR,   "ppo_doorkey.png")
WINDOW         = 100
# ────────────────────────────────────────────────────────────────────────────


def obs_to_tensor(obs: dict) -> torch.Tensor:
    img = obs["image"].flatten().astype(np.float32) / 10.0
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([img, direction]))


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(256, N_ACTIONS)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        feat   = self.shared(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value


def compute_gae(rewards, values, dones, last_val, gamma, lam):
    """Generalised Advantage Estimation."""
    advantages = []
    gae = 0.0
    values_ext = values + [last_val]           # bootstrap value appended
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values_ext[:-1])]
    return advantages, returns


def train():
    env      = gym.make(ENV_ID)
    obs0, _  = env.reset()
    state_dim = len(obs_to_tensor(obs0))

    model = ActorCritic(state_dim)
    opt   = optim.Adam(model.parameters(), lr=LR)

    obs, _          = env.reset()
    episode_rewards = []
    ep_reward       = 0.0
    ep_count        = 0

    while ep_count < N_EPISODES:
        # ── Collect rollout ──────────────────────────────────────────────────
        buf_states, buf_actions, buf_rewards, buf_dones    = [], [], [], []
        buf_log_probs, buf_values                          = [], []

        for _ in range(ROLLOUT_STEPS):
            state = obs_to_tensor(obs)
            with torch.no_grad():
                logits, val = model(state)
            dist   = Categorical(logits=logits)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buf_states.append(state)
            buf_actions.append(action)
            buf_rewards.append(reward)
            buf_dones.append(float(done))
            buf_log_probs.append(dist.log_prob(action).detach())
            buf_values.append(val.item())

            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_count += 1
                obs, _ = env.reset()

                if ep_count % 500 == 0:
                    avg = np.mean(episode_rewards[-100:])
                    print(f"  Episode {ep_count:5d}/{N_EPISODES}  avg100={avg:.4f}")

                if ep_count >= N_EPISODES:
                    break

        # ── Bootstrap last value ──────────────────────────────────────────────
        with torch.no_grad():
            _, last_val = model(obs_to_tensor(obs))
        last_val = last_val.item() * (1.0 - buf_dones[-1])

        # ── GAE advantages ────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_val, GAMMA, GAE_LAMBDA
        )

        # Convert to tensors
        states_t    = torch.stack(buf_states)
        actions_t   = torch.stack(buf_actions)
        old_lp_t    = torch.stack(buf_log_probs)
        returns_t   = torch.tensor(returns,    dtype=torch.float32)
        adv_t       = torch.tensor(advantages, dtype=torch.float32)
        adv_t       = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO optimisation epochs ───────────────────────────────────────────
        n = len(buf_states)
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(n)
            for start in range(0, n, MINI_BATCH):
                mb = idxs[start:start + MINI_BATCH]
                s  = states_t[mb]
                a  = actions_t[mb]
                r  = returns_t[mb]
                adv = adv_t[mb]
                old_lp = old_lp_t[mb]

                logits, values = model(s)
                dist   = Categorical(logits=logits)
                new_lp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio      = (new_lp - old_lp).exp()
                surr1      = ratio * adv
                surr2      = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.smooth_l1_loss(values, r)
                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

    env.close()
    return episode_rewards, model


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="chocolate", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="chocolate",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"PPO — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


def render_greedy(model: ActorCritic):
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    done   = False
    total  = 0.0
    with torch.no_grad():
        while not done:
            state  = obs_to_tensor(obs)
            logits, _ = model(state)
            action = logits.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += reward
    env.close()
    print(f"  Greedy episode reward: {total:.4f}")


# ── Entry-point ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=f"PPO on {ENV_ID}")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

print("=" * 55)
print(f"  PPO  |  {ENV_ID}")
print("=" * 55)

rewards, model = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

if args.render:
    render_greedy(model)
