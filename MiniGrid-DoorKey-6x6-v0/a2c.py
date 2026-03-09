"""
A2C (Synchronous Advantage Actor-Critic) on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

───────────────────────────── CORE INTUITION ─────────────────────────────
A2C = REINFORCE + a learned baseline (the critic) + bootstrapping.
REINFORCE has high variance because it uses the raw return G_t to weight the
policy gradient.  A2C fixes this by subtracting a "baseline" V(s) — the
critic's prediction of how good the state is.  The gradient now uses the
ADVANTAGE  A = R - V(s)  ("how much better than expected?") instead of the raw
return G, dramatically reducing variance.  It also bootstraps (uses V(s') to
estimate future returns after N steps instead of waiting for the episode to
end), which adds bias but further lowers variance and enables learning
mid-episode.  The actor (policy) and critic (value) share a network trunk.
Family: neural, model-free, on-policy, actor-critic, TD(n-step).

Relation to other algorithms:
  REINFORCE → (add value baseline = critic) → Actor-Critic
  Actor-Critic → (synchronous, advantage-based) → A2C
  A2C → (add clipped surrogate + minibatch epochs) → PPO
──────────────────────────────────────────────────────────────────────────

Observation: flattened partial-view image (7×7×3 = 147) + direction → 148 dims.
Actions    : 0-6 (full MiniGrid action set).
"""
#%%
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
N_EPISODES    = 5_000
N_STEPS       = 16            # rollout steps before each update
LR            = 3e-4
GAMMA         = 0.99
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.01
MAX_GRAD_NORM = 0.5
N_ACTIONS     = 7
ENV_ID        = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR   = "results"
PLOTS_DIR     = "plots"
RESULTS_FILE  = os.path.join(RESULTS_DIR, "a2c_doorkey_rewards.npy")
PLOT_FILE     = os.path.join(PLOTS_DIR,   "a2c_doorkey.png")
WINDOW        = 100
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
        features = self.shared(x)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)
        return logits, value


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
        states, actions, rewards, dones, values, log_probs, entropies = \
            [], [], [], [], [], [], []

        for _ in range(N_STEPS):
            state  = obs_to_tensor(obs)
            logits, val = model(state)
            dist   = Categorical(logits=logits)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(val)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_count += 1
                obs, _ = env.reset()

                # useful for printing states and undertstanding training dynamics when something goes wrong
                from IPython import embed
                ep_count==500 and embed(header=f"Reached episode {ep_count} — training loop debug breakpoint")

                if ep_count % 50 == 0:
                    avg = np.mean(episode_rewards[-100:])
                    print(f"  Episode {ep_count:5d}/{N_EPISODES}  avg100={avg:.4f}")

                if ep_count >= N_EPISODES:
                    break

        # ── Compute returns (bootstrap from last value) ──────────────────────
        with torch.no_grad():
            _, last_val = model(obs_to_tensor(obs))
        last_val = last_val.item() * (1.0 - dones[-1])

        returns = []
        G = last_val
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + GAMMA * G * (1.0 - d)
            returns.insert(0, G)

        returns_t    = torch.tensor(returns, dtype=torch.float32)
        values_t     = torch.stack(values)
        log_probs_t  = torch.stack(log_probs)
        entropies_t  = torch.stack(entropies)

        advantages = (returns_t - values_t.detach())

        # ── Update ──────────────────────────────────────────────────────────
        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.smooth_l1_loss(values_t, returns_t)
        entropy_loss = -ENTROPY_COEF * entropies_t.mean()
        loss = actor_loss + VALUE_COEF * critic_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()

    env.close()
    return episode_rewards, model


def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="mediumpurple", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="mediumpurple",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"A2C — {ENV_ID}")
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


# ── Run training immediately (no CLI) ───────────────────────────────────────
print("=" * 55)
print(f"  A2C  |  {ENV_ID}")
print("=" * 55)

rewards, model = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)


env = gym.make(ENV_ID, render_mode="human")
obs, _ = env.reset()
state = obs_to_tensor(obs)
logits, value = model(state)
dist = Categorical(logits=logits)
action = dist.sample().item()
next_obs, reward, terminated, truncated, _ = env.step(action)

print("\n--- post-training check ---")
print("initial obs:", obs)
print("state tensor:", state)
print("logits:", logits)
print("value estimate:", value)
print("sampled action:", action)
print("next_obs:", next_obs)
print("reward:", reward, "done:", terminated or truncated)

env.close()


# %%
