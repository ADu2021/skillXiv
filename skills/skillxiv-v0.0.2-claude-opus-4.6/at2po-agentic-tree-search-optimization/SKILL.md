---
name: at2po-agentic-tree-search-optimization
title: "AT²PO: Agentic Turn-based Policy Optimization via Tree Search"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.04767"
keywords: [Agent Reinforcement Learning, Tool Use, Tree Search, Multi-Hop Reasoning]
description: "Optimize multi-turn agent policies via entropy-guided tree expansion and turn-level credit assignment. AT²PO addresses exploration diversity, sparse credit signal, and policy misalignment problems in LLM agents through structured tree search and turn-aware policy updates."
---

## When to Use This Skill
- Multi-turn agent tasks requiring tool interaction (web search, knowledge bases)
- Multi-hop reasoning where information gathering spans multiple steps
- Sparse reward scenarios where only final task completion provides feedback
- Agents making discrete tool-calling decisions (HotpotQA, WikiQA, retrieval tasks)
- Environments with 3-6 tool calls per trajectory

## When NOT to Use This Skill
- Single-turn decision making (overkill for one-step tasks)
- Continuous action spaces (method designed for discrete tool calling)
- Scenarios with dense intermediate rewards (tree search provides marginal benefit)
- Real-time systems where planning overhead is prohibitive

## Problem Summary
Multi-turn agent reinforcement learning faces three critical challenges: (1) limited exploration diversity when policy entropy is low, (2) sparse credit assignment where only final success provides feedback (no signal for intermediate steps), and (3) misaligned policy optimization—token-level policy updates may not reflect the turn-level decisions agents actually make. These problems compound in multi-hop reasoning where agents must gather information across multiple tool calls before arriving at answers.

## Solution: AT²PO Tree Search + Turn-Level Policy Framework

Combine entropy-guided tree exploration with turn-aware policy optimization that aligns updates to actual agent decision structure.

```python
class AT2PO:
    def __init__(self, model, tree_depth=2, branching_factor=10):
        self.model = model
        self.tree_depth = tree_depth
        self.max_branches = branching_factor

    def entropy_guided_tree_expansion(self, root_state, num_iterations=2):
        """Expand tree from uncertain turns to promote diverse exploration"""
        expanded_nodes = []

        for iteration in range(num_iterations):
            # Score all leaf nodes by policy entropy
            leaf_scores = []
            for leaf in self.tree.leaves:
                action_logits = self.model(leaf.state)
                entropy = compute_entropy(action_logits)
                # Apply branching penalty to prevent over-expansion
                score = entropy - BRANCHING_PENALTY * leaf.depth
                leaf_scores.append((leaf, score))

            # Select K highest-entropy nodes (most uncertain)
            K = min(6, len(leaf_scores))
            selected_leaves = sorted(leaf_scores, key=lambda x: x[1], reverse=True)[:K]

            # Expand each selected leaf
            for leaf, _ in selected_leaves:
                # Sample M diverse continuations from high-entropy positions
                for _ in range(self.max_branches):
                    new_trajectory = self.sample_continuation(leaf)
                    expanded_nodes.append(new_trajectory)

        return expanded_nodes

    def turn_wise_credit_assignment(self, tree):
        """Compute node values via Monte Carlo bootstrapping"""
        node_values = {}

        # Bottom-up value propagation
        for node in reversed(tree.nodes):
            if node.is_leaf:
                # Terminal value: task success/failure
                node_values[node] = node.reward
            else:
                # Internal node value: entropy-weighted aggregate of descendants
                descendant_rewards = [
                    node_values[child] for child in node.children
                ]
                entropy_weights = [
                    compute_entropy(child.action_logits)
                    for child in node.children
                ]
                # Weighted average emphasizes uncertain paths
                node_values[node] = weighted_average(
                    descendant_rewards, entropy_weights
                )

        return node_values

    def turn_based_policy_optimization(self, trajectories, node_values):
        """Importance sampling + clipping at TURN level (not token level)"""
        turn_losses = []

        for trajectory in trajectories:
            for turn_idx, turn in enumerate(trajectory.turns):
                # Collect all tokens in this turn
                turn_tokens = turn.tokens
                turn_old_logprobs = turn.old_log_probs

                # Compute advantages at turn level
                turn_advantage = node_values[turn.end_node] - baseline(turn.start_node)

                # Apply clipping to prevent divergence
                turn_ratio = torch.exp(
                    turn.new_log_probs - turn_old_logprobs
                )
                clipped_ratio = torch.clamp(
                    turn_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS
                )

                # Minimize clipped surrogate loss at turn granularity
                turn_loss = -torch.min(
                    turn_ratio * turn_advantage,
                    clipped_ratio * turn_advantage
                )
                turn_losses.append(turn_loss)

        return torch.mean(torch.cat(turn_losses))
```

## Key Implementation Details

**Architecture & Training Configuration:**
- Backbone models: Qwen3-4B, Qwen3-8B, Qwen2.5-7B
- Datasets: Seven benchmarks including HotpotQA, 2WikiMultiHopQA, MuSiQue (multi-hop), NQ, TriviaQA, PopQA (single-hop)
- Training: 240 steps, batch size 64
- Tree parameters: M=10 initial branches, L=2 expansion iterations, K=6 nodes selected per iteration
- Maximum tool calls: 6 per trajectory

**Reward Function:**
Binary exact-match scoring with format validation constraints:

```python
def compute_turn_reward(trajectory, gold_answer):
    """Reward depends on task success and format compliance"""
    if trajectory.final_answer == gold_answer:
        # Bonus if reached answer efficiently
        efficiency_bonus = 1.0 - (num_turns / max_turns) * 0.2
        return 1.0 + efficiency_bonus
    elif trajectory.final_answer_format_valid():
        # Partial credit for correct format, wrong answer
        return 0.5
    else:
        return 0.0
```

**Entropy-Guided Expansion Rationale:**
Rather than uniform random branching, prioritize exploring turns where policy entropy is high (model uncertainty). This concentrates search effort on genuinely difficult decisions while skipping confident, low-uncertainty tokens.

## Performance Results

**Multi-Hop Reasoning (HotpotQA, 2WikiMultiHop, MuSiQue):**
- Improvement: +1.84 percentage points over state-of-the-art
- 43.5% fewer single-hop samples needed to match multi-hop performance
- Particularly strong on tasks requiring cross-document reasoning

**Single-Hop Tasks (NQ, TriviaQA, PopQA):**
- Consistent improvements (0.5-1.5 pp)
- Less dramatic than multi-hop (single-hop has denser feedback)

**Benchmark Average:**
- Seven-benchmark improvement: +1.84 pp
- Consistent gains across model sizes (4B to 8B)

## Advantages Over Baselines

- **vs. Token-Level RL**: Aligns policy updates to actual agent turns (multi-token decisions)
- **vs. Uniform Search**: Entropy guidance focuses exploration on uncertain decisions
- **vs. Dense Reward**: Credit assignment propagates final task outcome to intermediate steps
- **vs. Fixed-Length Plans**: Tree structure accommodates variable tool call sequences

## Implementation Checklist

1. **Prepare Environment**: Integrate tool APIs (search, database, retrieval)
2. **Dataset Preprocessing**: Label examples by difficulty (single-hop vs. multi-hop)
3. **Model Training**: Run SFT baseline on tool-use data
4. **Tree Search Setup**: Configure M, L, K parameters for your task
5. **RL Training**: Run AT²PO for 240+ steps with batch-wise updates
6. **Evaluation**: Measure task success rate + tool call efficiency
7. **Scaling**: Extend to larger models with adjusted tree parameters

## Reference Implementation
The authors released code supporting HotpotQA, 2WikiMultiHop, and BAMBOOGLE benchmarks with pre-trained checkpoints.
