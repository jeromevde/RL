# RL — Minimal Reinforcement Learning Implementations

A compact, educational repository covering six core reinforcement learning algorithms, all running on two tiny environments. Every file fits in one screen. Every equation is linked to the paper that introduced it.

```
CartPole-v1   →  DQN, PPO (discrete)
Pendulum-v1   →  PPO (continuous), DDPG, SAC, Behaviour Cloning, GAIL
```

---

## Algorithms at a Glance

| Algorithm | Type | Action Space | Key Paper |
|-----------|------|-------------|-----------|
| [DQN](#dqn--deep-q-network) | Model-free, off-policy | Discrete | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) |
| [PPO](#ppo--proximal-policy-optimization) | Model-free, on-policy | Both | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| [DDPG](#ddpg--deep-deterministic-policy-gradient) | Model-free, off-policy | Continuous | [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971) |
| [SAC](#sac--soft-actor-critic) | Model-free, off-policy | Continuous | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) |
| [BC](#bc--behaviour-cloning) | Imitation | Continuous | [Pomerleau, 1989](https://papers.nips.cc/paper_files/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) |
| [GAIL](#gail--generative-adversarial-imitation-learning) | Imitation (adversarial) | Continuous | [Ho & Ermon, 2016](https://arxiv.org/abs/1606.03476) |

---

## Quick Start

```bash
pip install -r requirements.txt

# Train a single algorithm
python train.py --algo dqn       # CartPole-v1
python train.py --algo ppo       # CartPole-v1 (discrete)
python train.py --algo ppo-c     # Pendulum-v1 (continuous)
python train.py --algo ddpg      # Pendulum-v1
python train.py --algo sac       # Pendulum-v1
python train.py --algo bc        # train SAC expert then clone with BC
python train.py --algo gail      # train SAC expert then GAIL

# Compare groups of algorithms
python train.py --compare discrete     # DQN vs PPO  on CartPole
python train.py --compare continuous   # DDPG vs SAC vs PPO on Pendulum
python train.py --compare imitation    # SAC expert vs BC vs GAIL
```

All training curves are saved to `plots/`.

---

## Project Structure

```
RL/
├── algorithms/
│   ├── dqn.py      # ~160 lines  — Q-network + target net + replay
│   ├── ppo.py      # ~200 lines  — clipped surrogate + GAE
│   ├── ddpg.py     # ~180 lines  — deterministic actor-critic + OU noise
│   ├── sac.py      # ~180 lines  — maximum entropy + twin critics + auto-α
│   ├── bc.py       # ~130 lines  — supervised cloning of expert demos
│   └── gail.py     # ~220 lines  — adversarial imitation (PPO + discriminator)
├── utils/
│   ├── networks.py  # MLP, QNetwork, Actors, Critics
│   ├── buffers.py   # ReplayBuffer (off-policy), RolloutBuffer (on-policy)
│   └── plotting.py  # Training curves, comparison plots
├── train.py         # CLI entry point
└── requirements.txt
```

---

## Environments

### CartPole-v1
A pole is attached to a cart. Push left or right to keep it balanced.

```
State:  [cart position, cart velocity, pole angle, pole angular velocity]  ∈ ℝ⁴
Action: {0=LEFT, 1=RIGHT}
Reward: +1 for every timestep the pole stays upright
Done:   pole angle > ±12° or cart position > ±2.4
```

Solved when mean reward ≥ 475 over 100 consecutive episodes.

### Pendulum-v1
Swing up a pendulum and keep it pointing straight up.

```
State:  [cos θ, sin θ, θ̇]  ∈ ℝ³
Action: torque ∈ [−2, 2]  (continuous)
Reward: −(θ² + 0.1 θ̇² + 0.001 a²)  (closer to vertical = higher reward)
```

Maximum reward per episode (200 steps) ≈ −200 (angle=0). Random policy ≈ −1200.

---

## DQN — Deep Q-Network

> *"Human-level control through deep reinforcement learning"*, Mnih et al., 2015  
> https://www.nature.com/articles/nature14236

**Core idea:** Approximate the optimal action-value function Q*(s,a) with a neural network.

### The Bellman Optimality Equation

```
Q*(s, a) = E[ r + γ max_{a'} Q*(s', a')  |  s, a ]
```

### DQN Loss

```
L(θ) = E_{(s,a,r,s') ~ D} [ ( r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ) )² ]
         ↑ sample from replay buffer              ↑ Bellman target (frozen)
```

### Two Key Tricks

| Trick | Why it helps |
|-------|-------------|
| **Experience replay** — store all (s,a,r,s') in buffer D, sample random mini-batches | Breaks temporal correlations; reuses past data |
| **Target network** θ⁻ — a frozen copy of θ, updated every C steps | Avoids "chasing a moving target" which causes training instability |

### ε-greedy Exploration

```
aₜ = random action           with probability ε
aₜ = argmax_a Q(sₜ, a; θ)   otherwise
```

ε is annealed from 1.0 → 0.01 over training.

**File:** [`algorithms/dqn.py`](algorithms/dqn.py)

---

## PPO — Proximal Policy Optimization

> *"Proximal Policy Optimization Algorithms"*, Schulman et al., 2017  
> https://arxiv.org/abs/1707.06347  
> *"High-Dimensional Continuous Control Using GAE"*, Schulman et al., 2016  
> https://arxiv.org/abs/1506.02438

**Core idea:** On-policy policy gradient with a clipped objective that prevents too-large updates.

### Policy Gradient Theorem

```
∇_θ J(θ) = E_τ [ Σ_t  ∇_θ log π_θ(aₜ|sₜ) · Âₜ ]
```

### PPO Clipped Objective

```
L_CLIP(θ) = E_t [ min( r_t(θ) · Âₜ ,  clip(r_t(θ), 1−ε, 1+ε) · Âₜ ) ]

r_t(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)    (probability ratio)
```

If `r_t` is outside [1−ε, 1+ε], the gradient is zeroed out — we ignore updates that move the policy too far.

### Generalised Advantage Estimation (GAE-λ)

```
δₜ  = rₜ + γ V(sₜ₊₁) − V(sₜ)                   (TD residual)
Âₜ  = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...         (GAE)
```

| λ | Bias | Variance | Intuition |
|---|------|----------|-----------|
| 0 | High | Low | One-step TD |
| 1 | Low | High | Monte Carlo |
| 0.95 | Balanced | Balanced | Recommended |

### Total Loss

```
L(θ, φ) = −L_CLIP(θ)  +  c₁ · (V_φ(sₜ) − Rₜ)²  −  c₂ · H[π_θ(·|sₜ)]
                 ↑ actor         ↑ critic loss           ↑ entropy bonus
```

**File:** [`algorithms/ppo.py`](algorithms/ppo.py)

---

## DDPG — Deep Deterministic Policy Gradient

> *"Continuous control with deep reinforcement learning"*, Lillicrap et al., 2015  
> https://arxiv.org/abs/1509.02971

**Core idea:** Actor-critic for continuous actions. The actor is *deterministic* — it directly outputs an action, no sampling needed.

### Deterministic Policy Gradient Theorem

```
∇_θ_μ J ≈ E_s [ ∇_a Q(s, a; θ_Q)|_{a=μ(s)} · ∇_θ_μ μ(s; θ_μ) ]
```

Chain rule: improve the actor by following the critic's gradient with respect to the action.

### Critic Update (Bellman error)

```
L(θ_Q) = E[ (y − Q(s,a;θ_Q))² ]
y       = r + γ Q(s', μ(s'; θ_μ⁻); θ_Q⁻)
```

### Soft Target Update (Polyak averaging)

```
θ⁻ ← τ·θ + (1−τ)·θ⁻      (τ ≪ 1, e.g. 0.005)
```

Gradual update is much more stable than hard copying every C steps.

### Exploration

```
aₜ = μ(sₜ; θ_μ) + Nₜ

dNₜ = θ(μ − Nₜ)dt + σ dWₜ   (Ornstein-Uhlenbeck process)
```

OU noise is temporally correlated, which works well for physical systems with inertia.

**File:** [`algorithms/ddpg.py`](algorithms/ddpg.py)

---

## SAC — Soft Actor-Critic

> *"Soft Actor-Critic"*, Haarnoja et al., 2018  
> https://arxiv.org/abs/1801.01290  
> *"SAC Algorithms and Applications"*, Haarnoja et al., 2018  
> https://arxiv.org/abs/1812.05905

**Core idea:** Maximum-entropy RL — maximise reward *and* entropy simultaneously. Exploration comes for free.

### Maximum Entropy Objective

```
J(π) = E_τ [ Σ_t  γᵗ ( r(sₜ, aₜ) + α · H(π(·|sₜ)) ) ]

H(π(·|s)) = −E_{a~π} [log π(a|s)]    (entropy)
α           = temperature (exploration vs exploitation)
```

### Twin Critics (reduces overestimation bias)

```
y = r + γ ( min(Q₁', Q₂')(s', ã') − α log π(ã'|s') )

L_Q = E[ (Q₁(s,a) − y)² ] + E[ (Q₂(s,a) − y)² ]
```

### Actor Update (reparameterisation trick)

```
ã = tanh(μ(s) + σ(s)·ε),    ε ~ N(0,I)    ← differentiable sample

L_π = E[ α log π(ã|s) − min_i Qᵢ(s, ã) ]
```

### Automatic Entropy Tuning

```
L(α) = E[ −α ( log π(a|s) + Ĥ ) ]    Ĥ = −|A|   (target entropy)
```

α is updated so the policy maintains entropy close to target Ĥ — no manual tuning needed.

**File:** [`algorithms/sac.py`](algorithms/sac.py)

---

## BC — Behaviour Cloning

> Pomerleau, 1989; Bain & Sammut, 1995

**Core idea:** Treat imitation as supervised learning. Given expert demonstrations D_E = {(sₜ, aₜ)}, minimise:

### Loss Functions

**Continuous actions (MSE):**
```
L_BC(θ) = E_{(s,a)~D_E} [ ‖μ_θ(s) − a‖² ]
```

**Discrete actions (cross-entropy):**
```
L_BC(θ) = −E_{(s,a)~D_E} [ log π_θ(a|s) ]
```

### Distributional Shift Problem

At test time, the cloned policy makes small mistakes that push it to states **never seen during training**. The expert's corrections were keeping it on track — without them, errors compound.

```
Expert:  s₀ → a₀ → s₁ → a₁ → s₂ → ...    (always on distribution)
Clone:   s₀ → â₀ → s₁'→ â₁'→ ...          (drifts off distribution)
```

**DAgger** (Ross et al., 2011) — Dataset Aggregation — fixes this:
1. Run current policy → collect states sₜ
2. Query expert for a*ₜ = π*(sₜ)
3. Add (sₜ, a*ₜ) to dataset and retrain

**File:** [`algorithms/bc.py`](algorithms/bc.py)

---

## GAIL — Generative Adversarial Imitation Learning

> *"Generative Adversarial Imitation Learning"*, Ho & Ermon, 2016  
> https://arxiv.org/abs/1606.03476

**Core idea:** Learn to imitate by fooling a discriminator — like a GAN, but the "generator" is a policy that interacts with the environment.

### Two-Player Game

```
min_π  max_{D_w}  E_{(s,a)~π_E}[log D_w(s,a)] + E_{(s,a)~π}[log(1 − D_w(s,a))]
```

### Discriminator Loss

```
L_D = −E_{expert}[log D_w(s,a)] − E_{policy}[log(1 − D_w(s,a))]
```

### Surrogate Reward for the Policy

```
r̃(s, a) = log D_w(s, a)     (higher = more "expert-like")
```

The policy is updated with PPO using r̃ instead of the true environment reward.

### Connection to IRL

GAIL implicitly performs Inverse Reinforcement Learning — the discriminator encodes a reward function under which the expert is near-optimal. Unlike classical IRL, no explicit reward extraction is needed.

### BC vs GAIL

| | BC | GAIL |
|--|----|----|
| Training data | Expert demos only | Expert demos + env interaction |
| Distributional shift | Severe | Avoided (RL explores its own states) |
| Training cost | Cheap (supervised) | Expensive (needs many env steps) |
| Stability | High | Lower (GAN dynamics) |

**File:** [`algorithms/gail.py`](algorithms/gail.py)

---

## Taxonomy

```
Reinforcement Learning
├── Model-Free
│   ├── On-Policy (uses only current policy's data)
│   │   └── PPO  ← stable, easy to tune; less sample efficient
│   └── Off-Policy (reuses past data from a replay buffer)
│       ├── Value-based (Q-learning)
│       │   └── DQN  ← discrete actions only
│       └── Actor-Critic
│           ├── DDPG  ← deterministic, sensitive to hyperparams
│           └── SAC   ← stochastic, automatic exploration, state-of-the-art
└── Imitation Learning (no environment reward signal needed)
    ├── Behaviour Cloning  ← supervised, fast, suffers distributional shift
    └── GAIL               ← adversarial, robust, needs env interaction
```

---

## Key Equations Summary

| Algorithm | Update Target | Exploration |
|-----------|--------------|-------------|
| DQN | `min(Q − y)²`, `y = r + γ max Q'` | ε-greedy |
| PPO | `min(r_t·Â, clip(r_t,1±ε)·Â)` | Stochastic policy |
| DDPG | DPG: `∇_a Q · ∇_μ` | OU noise |
| SAC | Entropy-augmented Bellman; twin critics | Entropy bonus (α·H) |
| BC | `‖μ_θ(s) − a_E‖²` | None (no RL) |
| GAIL | Discriminator reward `log D(s,a)` + PPO | PPO stochastic policy |

---

## Further Reading

- [Spinning Up in Deep RL](https://spinningup.openai.com/) — OpenAI's guide with algorithm implementations
- [Lilian Weng's blog — Policy Gradients](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — Policy gradient methods explained
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/) — Lecture slides and videos (UCL/DeepMind)
- [Sutton & Barto — RL: An Introduction](http://incompleteideas.net/book/the-book.html) — The canonical RL textbook (free PDF)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — Production-quality implementations of these same algorithms
