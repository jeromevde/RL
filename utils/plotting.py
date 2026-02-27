"""
Plotting utilities for training visualization.

Generates clean, publication-quality training curves so you can visually
compare algorithms at a glance.
"""

import matplotlib
matplotlib.use("Agg")  # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Consistent colour palette for all algorithms
ALGO_COLORS = {
    "DQN":   "#2196F3",   # blue
    "PPO":   "#4CAF50",   # green
    "DDPG":  "#FF9800",   # orange
    "SAC":   "#9C27B0",   # purple
    "BC":    "#F44336",   # red
    "GAIL":  "#009688",   # teal
}


def smooth(values, window=10):
    """Exponential moving average for smoother curves."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training(rewards, algo_name="Agent", save_path=None, window=10, title=None):
    """Plot a single training run: raw rewards + smoothed curve.

    Args:
        rewards:   list of episode rewards
        algo_name: label for the legend
        save_path: if given, saves figure to this path (e.g. 'dqn.png')
        window:    smoothing window size
        title:     optional plot title (defaults to '<algo_name> Training')
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    color = ALGO_COLORS.get(algo_name, "#333333")
    episodes = np.arange(1, len(rewards) + 1)

    ax.plot(episodes, rewards, alpha=0.25, color=color, linewidth=0.8)
    ax.plot(episodes, smooth(rewards, window), color=color, linewidth=2,
            label=f"{algo_name} (smoothed)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title or f"{algo_name} Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot → {save_path}")
    plt.close(fig)
    return fig


def compare_algorithms(results, save_path=None, title="Algorithm Comparison", window=10):
    """Overlay multiple training curves on one axes for easy comparison.

    Args:
        results:   dict  {algo_name: [episode_rewards]}
        save_path: if given, saves figure
        title:     plot title
        window:    smoothing window

    Example::

        compare_algorithms({
            "DQN":  dqn_rewards,
            "PPO":  ppo_rewards,
        }, save_path="comparison.png")
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, rewards in results.items():
        color = ALGO_COLORS.get(algo, None)
        episodes = np.arange(1, len(rewards) + 1)
        smoothed = smooth(rewards, window)
        ax.plot(episodes, rewards, alpha=0.15, color=color)
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=algo)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot → {save_path}")
    plt.close(fig)
    return fig


def plot_value_landscape(agent, env_name, save_path=None):
    """Visualise Q-values or policy logits across the state space (2D slice).

    Useful for CartPole: sweeps cart position × pole angle and plots Q-values.
    Works for DQN agents that implement a `.q_net` attribute.
    """
    import torch
    try:
        import gymnasium as gym
        env = gym.make(env_name)
    except Exception:
        return None

    # For CartPole: dim 0 = cart pos, dim 2 = pole angle
    pos = np.linspace(-2.4, 2.4, 40)
    ang = np.linspace(-0.2, 0.2, 40)
    P, A = np.meshgrid(pos, ang)
    q_left = np.zeros_like(P)

    for i in range(40):
        for j in range(40):
            s = np.array([P[i, j], 0.0, A[i, j], 0.0], dtype=np.float32)
            with torch.no_grad():
                q = agent.q_net(torch.FloatTensor(s).unsqueeze(0))
            q_left[i, j] = q[0, 0].item()

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(P, A, q_left, levels=20, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="Q(s, LEFT)")
    ax.set_xlabel("Cart position")
    ax.set_ylabel("Pole angle (rad)")
    ax.set_title("DQN Q-value landscape (action=LEFT)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot → {save_path}")
    plt.close(fig)
    return fig
