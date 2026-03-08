"""
TRPO (Trust Region Policy Optimization) on MiniGrid-DoorKey-6x6-v0
Standalone script — no shared dependencies.

Observation : flattened partial-view image (7×7×3 = 147) + direction → 148 dims.
Actions     : 0-6 (full MiniGrid action set, discrete).

Algorithm:
  1. Collect a full rollout of ROLLOUT_STEPS steps.
  2. Compute GAE advantages + normalise.
  3. Compute the policy gradient g.
  4. Compute the natural gradient direction s = F⁻¹g via Conjugate Gradient,
     where F is the Fisher Information Matrix approximated via Hessian-vector
     products of the KL divergence.
  5. Find the maximum step size δ satisfying the KL trust-region constraint.
  6. Apply a backtracking line search on the policy parameters.
  7. Update the value network with multiple gradient steps (separate from policy).

References:
  Schulman et al. (2015) "Trust Region Policy Optimization"
"""

import copy
import os

import gymnasium as gym
import minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ── Hyper-parameters ─────────────────────────────────────────────────────────
N_EPISODES       = 5_000
ROLLOUT_STEPS    = 256          # steps per policy update
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
MAX_KL           = 0.01         # KL trust-region constraint
DAMPING          = 0.1          # diagonal damping for CG (numerical stability)
CG_ITERS         = 10           # conjugate gradient iterations
LS_MAX_ITERS     = 10           # backtracking line search iterations
LS_ALPHA         = 0.5          # line search backtracking ratio
VALUE_LR         = 3e-4
VALUE_EPOCHS     = 5            # value network update steps per rollout
N_ACTIONS        = 7
ENV_ID           = "MiniGrid-DoorKey-6x6-v0"
RESULTS_DIR      = "results"
PLOTS_DIR        = "plots"
RESULTS_FILE     = os.path.join(RESULTS_DIR, "trpo_doorkey_rewards.npy")
PLOT_FILE        = os.path.join(PLOTS_DIR,   "trpo_doorkey.png")
WINDOW           = 100
# ─────────────────────────────────────────────────────────────────────────────


def obs_to_tensor(obs: dict) -> torch.Tensor:
    img       = obs["image"].flatten().astype(np.float32) / 10.0
    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([img, direction]))


# ── Networks ──────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits

    def get_dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── TRPO utilities ────────────────────────────────────────────────────────────

def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view(p.shape))
        idx += n


def flat_grad(loss: torch.Tensor, model: nn.Module,
              create_graph: bool = False) -> torch.Tensor:
    grads = torch.autograd.grad(loss, model.parameters(),
                                create_graph=create_graph, allow_unused=True)
    return torch.cat([
        g.view(-1) if g is not None else torch.zeros(p.numel())
        for g, p in zip(grads, model.parameters())
    ])


def mean_kl(policy: PolicyNet, states: torch.Tensor,
            old_probs: torch.Tensor) -> torch.Tensor:
    """KL(old || new) averaged over the batch."""
    new_probs = F.softmax(policy(states), dim=-1)
    # KL(P||Q) = Σ P log(P/Q)
    kl = (old_probs * (old_probs.log() - new_probs.log() + 1e-8)).sum(dim=1)
    return kl.mean()


def hessian_vector_product(policy: PolicyNet, states: torch.Tensor,
                            old_probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Computes F·v where F is the Fisher information matrix (Hessian of KL)."""
    kl   = mean_kl(policy, states, old_probs)
    g    = flat_grad(kl, policy, create_graph=True)
    Fv   = flat_grad((g * v.detach()).sum(), policy)
    return Fv + DAMPING * v          # Tikhonov damping


def conjugate_gradient(policy: PolicyNet, states: torch.Tensor,
                        old_probs: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve F·x = b approximately via CG."""
    x   = torch.zeros_like(b)
    r   = b.clone()
    p   = b.clone()
    rdot = r @ r

    for _ in range(CG_ITERS):
        Fp    = hessian_vector_product(policy, states, old_probs, p)
        alpha = rdot / (p @ Fp + 1e-8)
        x     = x + alpha * p
        r     = r - alpha * Fp
        rdot_new = r @ r
        beta  = rdot_new / (rdot + 1e-8)
        p     = r + beta * p
        rdot  = rdot_new
        if rdot < 1e-10:
            break
    return x


def compute_gae(rewards, values, dones, last_val, gamma, lam):
    advantages = []
    gae = 0.0
    values_ext = values + [last_val]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values_ext[:-1])]
    return advantages, returns


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    env      = gym.make(ENV_ID)
    obs0, _  = env.reset()
    state_dim = len(obs_to_tensor(obs0))

    policy    = PolicyNet(state_dim)
    value_net = ValueNet(state_dim)
    opt_value = optim.Adam(value_net.parameters(), lr=VALUE_LR)

    obs, _          = env.reset()
    episode_rewards = []
    ep_reward       = 0.0
    ep_count        = 0

    while ep_count < N_EPISODES:
        # ── Collect rollout ──────────────────────────────────────────────────
        buf_states, buf_actions, buf_rewards = [], [], []
        buf_dones,  buf_values              = [], []

        for _ in range(ROLLOUT_STEPS):
            state = obs_to_tensor(obs)
            with torch.no_grad():
                dist  = policy.get_dist(state)
                action = dist.sample()
                val    = value_net(state).item()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buf_states.append(state)
            buf_actions.append(action)
            buf_rewards.append(reward)
            buf_dones.append(float(done))
            buf_values.append(val)

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

        # ── Bootstrap + GAE ──────────────────────────────────────────────────
        with torch.no_grad():
            last_val = value_net(obs_to_tensor(obs)).item() * (1.0 - buf_dones[-1])

        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_val, GAMMA, GAE_LAMBDA)

        states_t  = torch.stack(buf_states)
        actions_t = torch.stack(buf_actions)
        returns_t = torch.tensor(returns,    dtype=torch.float32)
        adv_t     = torch.tensor(advantages, dtype=torch.float32)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── TRPO policy update ───────────────────────────────────────────────
        with torch.no_grad():
            old_dists = policy.get_dist(states_t)
            old_log_probs_a = old_dists.log_prob(actions_t)
            old_probs_all   = F.softmax(policy(states_t), dim=-1)  # for KL

        # Surrogate loss (policy gradient direction)
        def surrogate():
            new_log_probs_a = policy.get_dist(states_t).log_prob(actions_t)
            ratio = (new_log_probs_a - old_log_probs_a).exp()
            return (ratio * adv_t).mean()

        surr      = surrogate()
        g         = flat_grad(surr, policy)         # policy gradient

        # Natural gradient via CG
        ng = conjugate_gradient(policy, states_t, old_probs_all, g.detach())

        # Maximum step size satisfying KL constraint
        #   α* = sqrt(2 * δ / (ng^T F ng))
        Fng  = hessian_vector_product(policy, states_t, old_probs_all, ng)
        step_size = torch.sqrt(2 * MAX_KL / (ng @ Fng + 1e-8))
        step      = step_size * ng

        # Backtracking line search
        old_params = get_flat_params(policy)
        expected_improve = (g @ step).item()

        for i in range(LS_MAX_ITERS):
            new_params = old_params + (LS_ALPHA ** i) * step
            set_flat_params(policy, new_params)

            new_surr = surrogate().item()
            kl_val   = mean_kl(policy, states_t, old_probs_all).item()
            actual_improve = new_surr - surr.item()

            if kl_val <= MAX_KL and actual_improve > 0:
                break   # accept step
        else:
            # No acceptable step found — restore old params
            set_flat_params(policy, old_params)

        # ── Value network update ─────────────────────────────────────────────
        for _ in range(VALUE_EPOCHS):
            v_pred = value_net(states_t)
            v_loss = F.smooth_l1_loss(v_pred, returns_t)
            opt_value.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            opt_value.step()

    env.close()
    return episode_rewards, policy, value_net


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_rewards(rewards, save_path: str):
    smoothed = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")
    fig, ax  = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.2, color="goldenrod", label="episode reward")
    ax.plot(range(WINDOW - 1, len(rewards)), smoothed, color="goldenrod",
            linewidth=2, label=f"{WINDOW}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"TRPO — {ENV_ID}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved → {save_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 55)
print(f"  TRPO  |  {ENV_ID}")
print("=" * 55)

rewards, policy, value_net = train()

os.makedirs(RESULTS_DIR, exist_ok=True)
np.save(RESULTS_FILE, rewards)
print(f"  Results saved → {RESULTS_FILE}")

plot_rewards(rewards, PLOT_FILE)

# ── Post-training check ───────────────────────────────────────────────────────

SHOW_POST = True
if SHOW_POST:
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    state  = obs_to_tensor(obs)

    with torch.no_grad():
        logits = policy(state)
        probs  = F.softmax(logits, dim=-1)
        value  = value_net(state)

    action = probs.argmax().item()
    next_obs, reward, terminated, truncated, _ = env.step(action)

    print("\n--- post-training check ---")
    print("obs (image shape):", obs["image"].shape, "direction:", obs["direction"])
    print("logits:           ", logits.numpy().round(4))
    print("action probs:     ", probs.numpy().round(4))
    print("value estimate:    {:.4f}".format(value.item()))
    print("entropy of π:      {:.4f}".format(Categorical(probs=probs).entropy().item()))
    print("greedy action:    ", action)
    print("reward:           ", reward, "  done:", terminated or truncated)
    env.close()
