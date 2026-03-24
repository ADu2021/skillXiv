---
name: see-upo-sequence-level-rl
title: "SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06554"
keywords: [Agentic RL, Multi-turn Agents, Convergence Guarantees, Critic-Free, Backward Induction]
description: "Train multi-turn AI agents with convergence guarantees using sequential backward-induction updates, eliminating the need for separate critic networks while maintaining theoretical optimality. Use for long-horizon agentic reasoning where monotonic improvement and global optimality are required."
---

# SeeUPO: Backward-Induction RL for Multi-Turn Agents

Multi-turn agent training presents a fundamental challenge: existing RL methods cannot simultaneously achieve critic-free operation (no separate value function) and convergence guarantees in sequential decision settings. Standard on-policy methods like PPO apply advantage estimation at every timestep independently, but multi-turn interactions require coordinating updates across timesteps where later decisions depend on earlier ones.

SeeUPO solves this by reframing multi-turn interaction as sequential bandit problems and applying backward induction—a classical technique from game theory. By updating turns in reverse execution order (last turn → first turn), the method ensures each turn optimizes against the true optimal continuation value of subsequent turns, enabling global optimality convergence without a learned value function.

## Core Concept

Standard RL updates policy parameters simultaneously across all timesteps:
∇J = E[∇ log π(a_t|s_t) · Â_t]

This works well for single-step decisions but creates misalignment in sequential settings. When updating turn t, turns t+1...T have not yet been optimized, so the true continuation value is unknown. A critic attempts to approximate it, but critic errors compound across turns.

SeeUPO inverts this: update turns from T→1. When updating turn t:
- All turns t+1...T are already optimal (updated previously)
- The continuation value is known exactly: V_t^* = max_a E[R_{t:T}|a_t]
- Turn t optimizes against true optimal continuation, not estimated value

This sequential coordination enables monotonic improvement and convergence to globally optimal policies.

## Architecture Overview

- **Backward Induction Setup**: Structure agent trajectories as sequence of T turns, where each turn is a decision point
- **Per-Turn Policy**: Each turn has its own policy π_t(a|s_t); updates occur sequentially from t=T→1
- **GRAE Advantages**: Use Group Relative Advantage Estimation (critic-free) for per-turn advantage computation
- **Sequential Updates**: Update turn t using advantages computed relative to turn t's group rollouts, knowing turns t+1:T are optimal
- **Convergence Mechanism**: Backward induction ensures monotonic improvement; each update improves the full trajectory, not just isolated decisions

## Implementation

The implementation requires three components: trajectory structuring, backward-induction updates, and advantage estimation.

First, structure your multi-turn trajectories for backward induction:

```python
import torch
import torch.nn as nn

class MultiTurnTrajectory:
    """Represents a multi-turn agent trajectory amenable to backward-induction updates."""

    def __init__(self, states, actions, rewards, log_probs, num_turns):
        """
        Args:
            states: List of state observations at each turn [s_0, s_1, ..., s_T]
            actions: List of actions taken [a_0, a_1, ..., a_{T-1}]
            rewards: List of reward signals [r_0, r_1, ..., r_{T-1}]
            log_probs: Policy log probabilities [log π(a_0|s_0), ..., log π(a_{T-1}|s_{T-1})]
            num_turns: Total number of turns T
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs
        self.num_turns = num_turns
        self.returns = self._compute_returns()

    def _compute_returns(self):
        """Compute cumulative returns from each turn onward: R_t = r_t + γ·r_{t+1} + ..."""
        returns = []
        cumulative = 0.0
        for r in reversed(self.rewards):
            cumulative = r + 0.99 * cumulative
            returns.append(cumulative)
        return list(reversed(returns))

    def get_turn(self, t):
        """Get data for turn t."""
        return {
            'state': self.states[t],
            'action': self.actions[t],
            'log_prob': self.log_probs[t],
            'return': self.returns[t] if t < len(self.returns) else 0.0
        }
```

Next, implement backward-induction group-relative advantage estimation:

```python
class BackwardInductionOptimizer:
    """Performs backward-induction updates for multi-turn agent training."""

    def __init__(self, policies, optimizer, gamma=0.99):
        """
        Args:
            policies: List of policy networks [π_0, π_1, ..., π_{T-1}]
            optimizer: PyTorch optimizer
            gamma: Discount factor
        """
        self.policies = policies
        self.optimizer = optimizer
        self.gamma = gamma
        self.num_turns = len(policies)

    def compute_grae_advantages(self, batch_trajectories, turn_idx):
        """
        Compute Group Relative Advantage Estimates for turn t (critic-free).
        Args:
            batch_trajectories: List of MultiTurnTrajectory objects
            turn_idx: Current turn t to update (0 <= t < T)
        Returns:
            advantages: Advantage estimates [batch_size]
        """
        returns = torch.tensor([traj.get_turn(turn_idx)['return'] for traj in batch_trajectories])

        # GRAE: advantage = return - mean(return), divided by std for stability
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8
        advantages = (returns - mean_return) / std_return

        return advantages

    def backward_induction_step(self, batch_trajectories, turn_idx):
        """
        Single backward-induction update for turn t.
        Turns t+1...T are assumed already optimized.
        Args:
            batch_trajectories: List of trajectories
            turn_idx: Turn to update (from T-1 down to 0)
        Returns:
            loss: Scalar loss for this turn
        """
        # Compute advantages for this turn only
        advantages = self.compute_grae_advantages(batch_trajectories, turn_idx)

        # Gather log probabilities for this turn
        log_probs = torch.tensor([traj.get_turn(turn_idx)['log_prob'] for traj in batch_trajectories])

        # Policy gradient loss for turn t
        loss = -(log_probs * advantages.detach()).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, batch_trajectories):
        """
        Full backward-induction training epoch: update turns from T → 1.
        Args:
            batch_trajectories: List of trajectories for this epoch
        """
        losses_by_turn = {}

        # Update turns in REVERSE order (backward induction)
        for t in range(self.num_turns - 1, -1, -1):
            loss = self.backward_induction_step(batch_trajectories, t)
            losses_by_turn[f"turn_{t}"] = loss

            print(f"Turn {t}: loss = {loss:.4f}")

        return losses_by_turn
```

Finally, integrate into your multi-turn agent training loop:

```python
def train_multi_turn_agent(agent_policies, trajectories, num_epochs=10):
    """
    Full training loop for multi-turn agents using backward induction.
    Args:
        agent_policies: List of turn-specific policies
        trajectories: List of collected multi-turn trajectories
        num_epochs: Number of training epochs
    """
    optimizer = torch.optim.Adam(
        [p for policy in agent_policies for p in policy.parameters()],
        lr=1e-4
    )
    bi_optimizer = BackwardInductionOptimizer(agent_policies, optimizer)

    for epoch in range(num_epochs):
        # Collect trajectories (simplified; use actual environment interaction)
        # batch_trajectories = [collect_trajectory(agent_policies, env) for _ in range(batch_size)]

        # Backward induction training: updates turns T → 1
        losses = bi_optimizer.train_epoch(trajectories)

        # Monotonic improvement guarantee: overall trajectory return should improve
        avg_return = sum([traj.returns[0] for traj in trajectories]) / len(trajectories)
        print(f"Epoch {epoch}: avg_return = {avg_return:.4f}")

    return agent_policies
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Update order | T → 1 (backward) | Critical: always update in reverse turn order. Forward updates lose optimality guarantees. |
| Advantage normalization | Per-turn standardization | Compute mean/std within each turn's group rollouts, not globally. |
| Discount factor γ | 0.99 | Standard; tune per task if needed. |
| Group size per turn | 4-16 | Larger groups reduce advantage variance; smaller groups reduce computation. |
| Policy learning rate | 1e-4 to 1e-5 | Backward induction is stable; can use lower LR than standard RL. |

**When to Use**
- Training multi-turn agents where you want convergence guarantees
- Long-horizon reasoning tasks (10-100+ turns) where credit assignment matters
- When you don't have access to a good learned value function or want to eliminate critic overhead
- Tasks with clear per-turn success metrics (e.g., web navigation, dialogue agents)

**When NOT to Use**
- Single-turn decision making (standard RL is simpler and equally valid)
- Tasks where the optimal policy at turn t depends on unknown future turns (non-stationary)
- Very short horizons (T < 3) where the overhead isn't justified

**Common Pitfalls**
- Updating in forward order (t=0→T) breaks the method—guarantees require backward order
- Not standardizing advantages per-turn; using global statistics causes gradient instability
- Mixing with critic-based value functions defeats the purpose; trust the backward-induction mechanism
- Not accounting for environment stochasticity; if P(s_{t+1}|a_t) is uncertain, continuation value becomes stochastic

## Reference

See https://arxiv.org/abs/2602.06554 for the full paper, which formalizes backward induction as a heterogeneous-agent multi-agent bandit problem, proves global optimality convergence, and validates on complex agentic benchmarks like WebArena and real-time planning tasks.
