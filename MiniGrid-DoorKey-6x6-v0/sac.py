"""
SAC (Soft Actor-Critic) — Discrete action variant — on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

Observation : flattened partial-view image (7×7×3 = 147) + direction → 148 dims.
Actions     : 0-6 (full MiniGrid action set, discrete).

Discrete SAC differences vs. continuous SAC:
  - Actor outputs a softmax distribution over all actions (no reparameterisation trick).
  - Twin critics each output Q(s, a) for ALL actions simultaneously (N_ACTIONS outputs).
  - Entropy is computed analytically from π(a|s) — no sampling needed for the entropy term.
  - Temperature α can be auto-tuned toward a target entropy H* = -log(1/N_ACTIONS).

References:
  Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
"""

import os
import random
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Hyper-parameters ─────────────────────────────────────────────────────────
N_EPISODES         = 5_000
LR_ACTOR           = 3e-4
LR_CRITIC          = 3e-4
LR_ALPHA           = 3e-4           # temperature learning rate
GAMMA              = 0.99
TAU                = 0.005          # soft target update coefficient
BUFFER_SIZE        = 100_000
BATCH_SIZE         = 64
LEARN_START        = 1_000          # steps before first gradient update
TARGET_ENTROPY     = -np.log(1.0 / 7) * 0.98  # slightly below max entropy
INIT_ALPHA         = 0.2
N_ACTIONS          = 7
ENV_ID             = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR        = "results"
PLOTS_DIR          = "plots"
RESULTS_FILE       = os.path.join(RESULTS_DIR, "sac_doorkey_rewards.npy")
PLOT_FILE          = os.path.join(PLOTS_DIR,   "sac_doorkey.png")
WINDOW             = 100
# ─────────────────────────────────────────────────────────────────────────────


def obs_to_np(obs: dict) -> np.ndarray:
    img = obs["image"].flatten().astype(np.float32) / 10.0
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
    return np.concatenate([img, direction])


# ── Networks ──────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Outputs a softmax action distribution π(a|s)."""
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        probs  = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, x: torch.Tensor):
        """Return sampled action + its log-prob + full distribution probs."""
        probs, log_probs = self.forward(x)
        dist   = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1), probs, log_probs


class Critic(nn.Module):
    """Twin Q-network — outputs Q(s, a) for ALL actions at once."""
    def __init__(self, state_dim: int):
        super().__init__()
        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),       nn.ReLU(),
                nn.Linear(256, N_ACTIONS),
            )
        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, x: torch.Tensor):
        return self.q1(x), self.q2(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

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


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    env      = gym.make(ENV_ID)
    obs0, _  = env.reset()
    state_dim = len(obs_to_np(obs0))

    actor         = Actor(state_dim)
    critic        = Critic(state_dim)
    critic_target = Critic(state_dim)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, dtype=torch.float32)
    alpha     = log_alpha.exp().item()

    opt_actor  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    opt_alpha  = optim.Adam([log_alpha],          lr=LR_ALPHA)

    buf       = ReplayBuffer(BUFFER_SIZE)
    step      = 0
    ep_rewards = []

    for ep in range(N_EPISODES):
        obs, _       = env.reset()
        state        = obs_to_np(obs)
        total_reward = 0.0
        done         = False

        while not done:
            alpha = log_alpha.exp().item()

            if step < LEARN_START:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    s_t    = torch.from_numpy(state).unsqueeze(0)
                    act_t, _, _, _ = actor.sample(s_t)
                action = act_t.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = obs_to_np(obs)

            buf.push(state, action, reward, next_state, float(done))
            state        = next_state
            total_reward += reward
            step         += 1

            if step >= LEARN_START and len(buf) >= BATCH_SIZE:
                s, a, r, s2, d = buf.sample(BATCH_SIZE)

                # ── Critic loss ──────────────────────────────────────────────
                with torch.no_grad():
                    probs2, log_probs2 = actor.forward(s2)
                    q1_t, q2_t        = critic_target(s2)
                    min_q2            = torch.min(q1_t, q2_t)
                    # Expected value over all actions (discrete SAC)
                    v_next = (probs2 * (min_q2 - alpha * log_probs2)).sum(dim=1)
                    q_target = r + GAMMA * v_next * (1.0 - d)

                q1, q2 = critic(s)
                q1_a   = q1.gather(1, a.unsqueeze(1)).squeeze(1)
                q2_a   = q2.gather(1, a.unsqueeze(1)).squeeze(1)
                critic_loss = F.smooth_l1_loss(q1_a, q_target) + \
                              F.smooth_l1_loss(q2_a, q_target)

                opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_critic.step()

                # ── Actor loss ───────────────────────────────────────────────
                probs, log_probs = actor.forward(s)
                with torch.no_grad():
                    q1_pi, q2_pi = critic(s)
                    min_q_pi     = torch.min(q1_pi, q2_pi)
                # Expectation over actions — no reparameterisation needed
                actor_loss = (probs * (alpha * log_probs - min_q_pi)).sum(dim=1).mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_actor.step()

                # ── Temperature loss ─────────────────────────────────────────
                # Auto-tune α so that H[π] ≈ TARGET_ENTROPY
                entropy_est  = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
                alpha_loss   = log_alpha * (entropy_est - TARGET_ENTROPY)

                opt_alpha.zero_grad()
                alpha_loss.backward()
                opt_alpha.step()

                # ── Soft target update ───────────────────────────────────────
                soft_update(critic, critic_target, TAU)

        ep_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(ep_rewards[-100:])
            print(f"  Episode {ep+1:5d}/{N_EPISODES}  step={step:7d}  α={alpha:.4f}  avg100={avg:.4f}")

    env.close()
    return ep_rewards, actor


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax  = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="teal", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="teal",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"SAC — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 55)
print(f"  SAC  |  {ENV_ID}")
print("=" * 55)

rewards, actor = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

# ── Post-training check ───────────────────────────────────────────────────────

SHOW_POST = True
if SHOW_POST:
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    state  = torch.from_numpy(obs_to_np(obs)).unsqueeze(0)

    with torch.no_grad():
        probs, log_probs = actor.forward(state)

    action = probs.argmax(dim=1).item()
    next_obs, reward, terminated, truncated, _ = env.step(action)

    print("\n--- post-training check ---")
    print("obs (image shape):", obs["image"].shape, "direction:", obs["direction"])
    print("action probs:     ", probs.squeeze().numpy().round(4))
    print("log probs:        ", log_probs.squeeze().numpy().round(4))
    print("entropy of π:      {:.4f}".format(-(probs * log_probs).sum().item()))
    print("greedy action:    ", action)
    print("reward:           ", reward, "  done:", terminated or truncated)
    env.close()
