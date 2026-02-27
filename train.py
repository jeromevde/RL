"""
train.py — Main training script
================================
Run individual algorithms or compare all of them side-by-side.

Usage
-----
# Train a single algorithm
python train.py --algo dqn
python train.py --algo ppo
python train.py --algo ddpg
python train.py --algo sac
python train.py --algo bc      # trains SAC expert then clones
python train.py --algo gail    # trains SAC expert then runs GAIL

# Compare discrete algorithms (CartPole)
python train.py --compare discrete

# Compare continuous algorithms (Pendulum)
python train.py --compare continuous

# Compare all imitation learning methods
python train.py --compare imitation
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ------------------------------------------------------------------
# Environment specs
# ------------------------------------------------------------------

DISCRETE_ENV  = "CartPole-v1"    # state: 4D, actions: 2 (LEFT / RIGHT)
CONTINUOUS_ENV = "Pendulum-v1"   # state: 3D, action: 1D torque ∈ [−2, 2]

DISCRETE_STATE_DIM   = 4
DISCRETE_N_ACTIONS   = 2

CONTINUOUS_STATE_DIM = 3
CONTINUOUS_ACTION_DIM = 1
CONTINUOUS_MAX_ACTION = 2.0

PLOTS_DIR = "plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Individual trainers
# ------------------------------------------------------------------

def train_dqn(n_episodes=400):
    from algorithms.dqn import DQN
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training DQN on", DISCRETE_ENV)
    print("="*60)

    agent = DQN(DISCRETE_STATE_DIM, DISCRETE_N_ACTIONS)
    rewards = agent.train(DISCRETE_ENV, n_episodes=n_episodes)

    ensure_plots_dir()
    plot_training(rewards, "DQN", save_path=f"{PLOTS_DIR}/dqn.png")
    return rewards, agent


def train_ppo_discrete(n_episodes=400):
    from algorithms.ppo import PPO
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training PPO (discrete) on", DISCRETE_ENV)
    print("="*60)

    agent = PPO(DISCRETE_STATE_DIM, DISCRETE_N_ACTIONS, continuous=False)
    rewards = agent.train(DISCRETE_ENV, n_episodes=n_episodes)

    ensure_plots_dir()
    plot_training(rewards, "PPO", save_path=f"{PLOTS_DIR}/ppo_discrete.png")
    return rewards, agent


def train_ppo_continuous(n_episodes=200):
    from algorithms.ppo import PPO
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training PPO (continuous) on", CONTINUOUS_ENV)
    print("="*60)

    agent = PPO(
        CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM,
        continuous=True, rollout_steps=1024,
    )
    rewards = agent.train(CONTINUOUS_ENV, n_episodes=n_episodes)

    ensure_plots_dir()
    plot_training(rewards, "PPO", save_path=f"{PLOTS_DIR}/ppo_continuous.png")
    return rewards, agent


def train_ddpg(n_episodes=200):
    from algorithms.ddpg import DDPG
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training DDPG on", CONTINUOUS_ENV)
    print("="*60)

    agent = DDPG(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    rewards = agent.train(CONTINUOUS_ENV, n_episodes=n_episodes)

    ensure_plots_dir()
    plot_training(rewards, "DDPG", save_path=f"{PLOTS_DIR}/ddpg.png")
    return rewards, agent


def train_sac(n_episodes=200):
    from algorithms.sac import SAC
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training SAC on", CONTINUOUS_ENV)
    print("="*60)

    agent = SAC(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    rewards = agent.train(CONTINUOUS_ENV, n_episodes=n_episodes)

    ensure_plots_dir()
    plot_training(rewards, "SAC", save_path=f"{PLOTS_DIR}/sac.png")
    return rewards, agent


def train_bc(n_expert_episodes=30, n_bc_epochs=100, n_eval_episodes=20):
    """Train a SAC expert, collect demos, then clone with BC."""
    from algorithms.sac import SAC
    from algorithms.bc import BehaviourCloning
    from utils.plotting import plot_training, compare_algorithms

    print("\n" + "="*60)
    print("  Training SAC expert for BC demo collection...")
    print("="*60)

    expert = SAC(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    expert.train(CONTINUOUS_ENV, n_episodes=200)

    print("\n  Collecting expert demonstrations...")
    bc_agent = BehaviourCloning(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    states, actions = bc_agent.collect_expert_demos(expert, CONTINUOUS_ENV, n_episodes=n_expert_episodes)
    print(f"  Collected {len(states)} (state, action) pairs from {n_expert_episodes} episodes.")

    print("\n  Cloning expert behaviour with supervised learning...")
    bc_agent.fit(states, actions, n_epochs=n_bc_epochs)

    print("\n  Evaluating cloned policy...")
    bc_rewards = bc_agent.evaluate(CONTINUOUS_ENV, n_episodes=n_eval_episodes)

    print(f"\n  BC mean reward:     {np.mean(bc_rewards):.1f} ± {np.std(bc_rewards):.1f}")

    ensure_plots_dir()
    plot_training(bc_rewards, "BC", save_path=f"{PLOTS_DIR}/bc_eval.png",
                  title="Behaviour Cloning — Evaluation Episodes")
    return bc_rewards, bc_agent


def train_gail(n_expert_episodes=30, n_gail_episodes=200):
    """Train a SAC expert, collect demos, then imitate with GAIL."""
    from algorithms.sac import SAC
    from algorithms.bc import BehaviourCloning
    from algorithms.gail import GAIL
    from utils.plotting import plot_training

    print("\n" + "="*60)
    print("  Training SAC expert for GAIL demo collection...")
    print("="*60)

    expert = SAC(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    expert.train(CONTINUOUS_ENV, n_episodes=200)

    # Reuse BC's collect utility
    helper = BehaviourCloning(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    states, actions = helper.collect_expert_demos(expert, CONTINUOUS_ENV, n_episodes=n_expert_episodes)
    print(f"  Collected {len(states)} expert (s,a) pairs.")

    print("\n  Running GAIL training...")
    agent = GAIL(
        CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM,
        expert_states=states, expert_actions=actions,
        max_action=CONTINUOUS_MAX_ACTION,
        ppo_kwargs={"rollout_steps": 512},
    )
    disc_rewards = agent.train(CONTINUOUS_ENV, n_episodes=n_gail_episodes)

    ensure_plots_dir()
    plot_training(disc_rewards, "GAIL", save_path=f"{PLOTS_DIR}/gail.png",
                  title="GAIL Training (Discriminator Reward)")
    return disc_rewards, agent


# ------------------------------------------------------------------
# Comparison runners
# ------------------------------------------------------------------

def compare_discrete(n_episodes=400):
    from utils.plotting import compare_algorithms

    results = {}
    results["DQN"], _ = train_dqn(n_episodes)
    results["PPO"], _ = train_ppo_discrete(n_episodes)

    ensure_plots_dir()
    compare_algorithms(results, save_path=f"{PLOTS_DIR}/compare_discrete.png",
                       title=f"Discrete Algorithms — {DISCRETE_ENV}")
    print(f"\nComparison plot saved to {PLOTS_DIR}/compare_discrete.png")


def compare_continuous(n_episodes=200):
    from utils.plotting import compare_algorithms

    results = {}
    results["DDPG"], _ = train_ddpg(n_episodes)
    results["SAC"],  _ = train_sac(n_episodes)
    results["PPO"],  _ = train_ppo_continuous(n_episodes)

    ensure_plots_dir()
    compare_algorithms(results, save_path=f"{PLOTS_DIR}/compare_continuous.png",
                       title=f"Continuous Algorithms — {CONTINUOUS_ENV}")
    print(f"\nComparison plot saved to {PLOTS_DIR}/compare_continuous.png")


def compare_imitation(n_gail_episodes=200, n_eval_episodes=20):
    """Compare SAC expert vs BC clone vs GAIL agent on Pendulum."""
    from algorithms.sac import SAC
    from algorithms.bc import BehaviourCloning
    from algorithms.gail import GAIL
    from utils.plotting import compare_algorithms

    print("\n" + "="*60)
    print("  Training SAC expert...")
    print("="*60)
    expert = SAC(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    expert_ep_rewards = expert.train(CONTINUOUS_ENV, n_episodes=200)

    helper = BehaviourCloning(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    states, actions = helper.collect_expert_demos(expert, CONTINUOUS_ENV, n_episodes=30)

    print("\n  Training BC clone...")
    bc_agent = BehaviourCloning(CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM, CONTINUOUS_MAX_ACTION)
    bc_agent.fit(states, actions, n_epochs=100)
    bc_rewards = bc_agent.evaluate(CONTINUOUS_ENV, n_episodes=n_eval_episodes)

    print("\n  Training GAIL agent...")
    gail_agent = GAIL(
        CONTINUOUS_STATE_DIM, CONTINUOUS_ACTION_DIM,
        expert_states=states, expert_actions=actions,
        max_action=CONTINUOUS_MAX_ACTION,
        ppo_kwargs={"rollout_steps": 512},
    )
    gail_agent.train(CONTINUOUS_ENV, n_episodes=n_gail_episodes)
    # Evaluate GAIL agent in the real environment
    import gymnasium as gym
    env = gym.make(CONTINUOUS_ENV)
    gail_rewards = []
    for _ in range(n_eval_episodes):
        s, _ = env.reset()
        ep_r = 0
        for _ in range(200):
            a = gail_agent.select_action(s)
            s, r, terminated, truncated, _ = env.step(a)
            ep_r += r
            if terminated or truncated:
                break
        gail_rewards.append(ep_r)
    env.close()

    results = {
        "SAC (expert)": expert_ep_rewards[-n_eval_episodes:],
        "BC":           bc_rewards,
        "GAIL":         gail_rewards,
    }
    ensure_plots_dir()
    compare_algorithms(results, save_path=f"{PLOTS_DIR}/compare_imitation.png",
                       title=f"Imitation Learning — {CONTINUOUS_ENV}")
    print(f"\nComparison plot saved to {PLOTS_DIR}/compare_imitation.png")
    print(f"SAC  mean reward: {np.mean(expert_ep_rewards[-n_eval_episodes:]):.1f}")
    print(f"BC   mean reward: {np.mean(bc_rewards):.1f}")
    print(f"GAIL mean reward: {np.mean(gail_rewards):.1f}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Algorithm Trainer")
    parser.add_argument("--algo", choices=["dqn", "ppo", "ppo-c", "ddpg", "sac", "bc", "gail"],
                        help="Single algorithm to train")
    parser.add_argument("--compare", choices=["discrete", "continuous", "imitation"],
                        help="Compare a group of algorithms")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    args = parser.parse_args()

    if args.algo:
        ep = args.episodes
        if args.algo == "dqn":
            train_dqn(ep or 400)
        elif args.algo == "ppo":
            train_ppo_discrete(ep or 400)
        elif args.algo == "ppo-c":
            train_ppo_continuous(ep or 200)
        elif args.algo == "ddpg":
            train_ddpg(ep or 200)
        elif args.algo == "sac":
            train_sac(ep or 200)
        elif args.algo == "bc":
            train_bc()
        elif args.algo == "gail":
            train_gail()
    elif args.compare:
        if args.compare == "discrete":
            compare_discrete(args.episodes or 400)
        elif args.compare == "continuous":
            compare_continuous(args.episodes or 200)
        elif args.compare == "imitation":
            compare_imitation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
