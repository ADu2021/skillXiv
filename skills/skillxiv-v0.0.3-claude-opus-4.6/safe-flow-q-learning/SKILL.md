---
name: safe-flow-q-learning
title: "Safe Flow Q-Learning: Reachability-Based Safe Reinforcement Learning with Flow-Matching Policies"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15136"
keywords: [Safe Reinforcement Learning, Reachability Analysis, Flow Matching, Offline RL, Safety Constraints]
description: "Train offline safe RL agents using Hamilton-Jacobi reachability principles to learn feasibility-gated policies. Combine reward and safety critics with flow-matching teacher policies, distill to one-step actors, and calibrate safety thresholds via conformal prediction—achieving near-zero constraint violations with 2.5× inference speedup."
---

# Safe Flow Q-Learning: Reachability-Based Safe RL

## Problem Statement

Offline safe RL methods using Lagrangian penalty approaches create conflicting gradients: reward maximization and safety recovery compete, requiring careful tuning. Diffusion-based policies guarantee safety but suffer from slow inference (multiple denoising steps). We need a method that prioritizes safety without soft penalties and runs efficiently at deployment.

## Component Innovation: Feasibility-Gated Objective

**The Modification:** Replace soft Lagrangian penalties with hard constraint masking that completely gates reward updates when actions violate safety constraints.

**Four-Stage Training Pipeline:**

1. **Critic Learning:** Train separate reward and safety critics using max-backup Bellman recursion inspired by Hamilton-Jacobi (HJ) reachability. Safety values propagate worst-case constraints backward through time.

2. **Flow Teacher:** Train a multi-step flow-matching policy that maps states to action distributions while respecting learned reachability constraints.

3. **Actor Distillation:** Distill multi-step flow model into a one-step deterministic actor μ_ω(x,z) that maps (state, noise) pairs directly to actions without integration.

4. **Conformal Calibration:** Use conformal prediction to adjust safety thresholds, accounting for finite-data approximation errors and providing probabilistic safety coverage.

**Feasibility-Gated Loss:**

```python
# Gated objective separates reward and safety recovery
# When predicted action satisfies reachability constraints:
#   L = E[Q_reward(s,a)]  # maximize reward
# When action violates constraints:
#   L = E[Q_safety(s,a)]  # recover feasibility only
# Binary mask prevents gradient conflicts between objectives
```

## Ablation & Safety-Performance Tradeoff

**Constraint Violations:** Near-zero violations across boat navigation and Safety Gymnasium MuJoCo environments, maintaining safety even in distributional shift scenarios.

**Reward Performance:** Competitive rewards compared to baselines (FISOR, C2IQL, CPQ) while achieving stringent safety requirements.

**Inference Speed:** 2.5× faster than diffusion-based alternatives by eliminating iterative denoising and rejection sampling.

**Key Tradeoff:** Hard masking may produce non-smooth loss landscapes; soft relaxations being explored for future work.

## Drop-In Checklist

1. **Offline Data:** Collect safe trajectories; ensure constraint labels are accurate
2. **Critic Initialization:** Pre-train reward and safety critics separately using standard Q-learning objectives
3. **Flow Model:** Train teacher using flow-matching loss with constraint awareness
4. **Actor Distillation:** Use KL divergence to match actor to teacher; verify one-step inference runs at target latency
5. **Conformal Calibration:** Compute prediction intervals on held-out validation set; adjust thresholds to achieve desired safety probability
6. **Test Safety:** Verify near-zero violations on test trajectories; accept marginal reward loss if constraint satisfaction ≥ 99%

## Conditions for Effectiveness

- **Constraint Definition:** Clear, deterministic constraint functions (e.g., position bounds, acceleration limits) work best; probabilistic constraints require careful calibration
- **Offline Data Quality:** Requires sufficient diversity to learn both reward and safety landscapes; very constrained offline data may underestimate reachable regions
- **Safety Margin:** Conformal prediction works best with held-out validation set ≥10% of offline data
- **Deployment Environment:** One-step actor assumes action execution is instantaneous; may need integration checks if action latency is significant
- **Horizon Length:** Tested on short-horizon tasks (50-200 steps); very long horizons may require re-calibration of safety thresholds

## Practical Implications

- **Safety-First Design:** Hard gating prioritizes constraint satisfaction over reward optimization—appropriate for safety-critical applications
- **Efficient Deployment:** One-step actors enable real-time control without computational overhead
- **Robustness:** Conformal calibration provides formal probability of safety violations, suitable for certified deployment
