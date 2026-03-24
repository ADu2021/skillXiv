---
name: matchtir-fine-grained-tool-reasoning
title: "MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.10712"
keywords: [tool-integration, credit-assignment, reinforcement-learning, bipartite-matching, long-horizon-reasoning]
description: "Improves tool-integrated reasoning by using bipartite matching to assign dense turn-level rewards, enabling credit assignment for individual tool interactions in multi-turn tasks where 4B models outperform 8B competitors."
---

## Overview

Enable agents to reason effectively with tools by providing fine-grained reward signals at each tool-interaction step. Rather than assigning uniform rewards across entire trajectories, use bipartite matching to align predicted tool interactions with ground-truth sequences, enabling dense per-turn credit assignment.

## When to Use

- For agents that invoke external tools across multiple turns
- For long-horizon tasks requiring many tool interactions
- When you want to improve credit assignment for tool use
- For training smaller models to outperform larger baselines

## When NOT to Use

- For single-turn tool invocations
- When ground-truth tool sequences are unavailable
- For fast-adapting environments where re-training is infeasible
- When computational cost of bipartite matching is prohibitive

## Key Technical Components

### Bipartite Matching Framework

Match predicted tool interactions to ground-truth sequences.

```python
# Bipartite matching for credit assignment
import numpy as np
from scipy.optimize import linear_sum_assignment

class BipartiteToolMatcher:
    def __init__(self):
        self.similarity_fn = self.compute_interaction_similarity

    def match_sequences(self, predicted_sequence, ground_truth_sequence):
        """Match predicted to ground-truth tool interactions"""
        # Create cost matrix
        cost_matrix = np.zeros((len(predicted_sequence), len(ground_truth_sequence)))

        for i, pred in enumerate(predicted_sequence):
            for j, truth in enumerate(ground_truth_sequence):
                cost_matrix[i][j] = -self.similarity_fn(pred, truth)

        # Solve assignment problem
        pred_indices, truth_indices = linear_sum_assignment(cost_matrix)

        # Return matching with confidence scores
        matches = []
        for p_idx, t_idx in zip(pred_indices, truth_indices):
            similarity = self.similarity_fn(
                predicted_sequence[p_idx],
                ground_truth_sequence[t_idx]
            )
            matches.append({
                "predicted_idx": p_idx,
                "truth_idx": t_idx,
                "similarity": similarity,
                "is_correct": similarity > THRESHOLD
            })

        return matches

    def compute_interaction_similarity(self, predicted, ground_truth):
        """Compute similarity between tool interactions"""
        # Compare tool name, arguments, output
        tool_match = 1.0 if predicted["tool"] == ground_truth["tool"] else 0.0
        arg_match = self.compare_arguments(predicted["args"], ground_truth["args"])
        output_match = self.compare_outputs(predicted["output"], ground_truth["output"])

        # Weighted combination
        similarity = 0.4 * tool_match + 0.3 * arg_match + 0.3 * output_match
        return similarity

    def compare_arguments(self, pred_args, truth_args):
        """Compare tool arguments with tolerance for minor variations"""
        # Exact match is ideal, but some flexibility for equivalent forms
        if pred_args == truth_args:
            return 1.0

        # Check semantic equivalence
        if self.semantically_equivalent(pred_args, truth_args):
            return 0.8

        # Partial match
        matching_keys = sum(1 for k in pred_args if k in truth_args and pred_args[k] == truth_args[k])
        return matching_keys / max(len(pred_args), len(truth_args))

    def compare_outputs(self, pred_output, truth_output):
        """Compare tool outputs"""
        if pred_output == truth_output:
            return 1.0
        # Allow numerical tolerance for close values
        try:
            if isinstance(pred_output, (int, float)) and isinstance(truth_output, (int, float)):
                rel_error = abs(pred_output - truth_output) / max(abs(truth_output), 1.0)
                return max(0.0, 1.0 - rel_error)
        except:
            pass
        return 0.0
```

### Turn-Level Advantage Estimation

Compute distinct advantages for each interaction turn.

```python
# Turn-level advantage computation
class TurnLevelAdvantage:
    def compute_turn_advantages(self, matches, trajectory_reward):
        """Assign advantage values to individual turns"""
        advantages = [0.0] * len(matches)

        for match in matches:
            if match["is_correct"]:
                # Correct interaction contributes to final reward
                advantages[match["predicted_idx"]] = trajectory_reward / len(matches)
            else:
                # Incorrect interaction incurs penalty
                advantages[match["predicted_idx"]] = -PENALTY_SCALE

        # Normalize advantages
        mean_adv = np.mean([a for a in advantages if a != 0.0])
        std_adv = np.std([a for a in advantages if a != 0.0]) + 1e-8

        normalized = [
            (a - mean_adv) / std_adv if a != 0.0 else 0.0
            for a in advantages
        ]

        return normalized

    def compute_dual_level_advantages(self, turn_advantages, trajectory_reward):
        """Combine turn-level and trajectory-level signals"""
        # Turn-level advantages guide step selection
        # Trajectory-level advantage ensures global optimization

        trajectory_advantage = trajectory_reward

        # Dual-level signal: local (turn) + global (trajectory)
        dual_advantages = []
        for turn_adv in turn_advantages:
            # Balance local and global signals
            combined = 0.7 * turn_adv + 0.3 * (trajectory_advantage / len(turn_advantages))
            dual_advantages.append(combined)

        return dual_advantages
```

### Multi-Turn Trajectory Processing

Handle variable-length tool interaction sequences.

```python
# Multi-turn trajectory handling
class MultiTurnTrajectory:
    def __init__(self, trajectory):
        self.turns = trajectory  # List of tool invocations
        self.length = len(trajectory)

    def extract_interactions(self):
        """Extract tool interactions from trajectory"""
        interactions = []
        for turn in self.turns:
            interaction = {
                "tool": turn["tool_name"],
                "args": turn["tool_args"],
                "output": turn["tool_output"],
                "success": turn["success"]
            }
            interactions.append(interaction)
        return interactions

    def compute_trajectory_reward(self, success_score, efficiency_score=1.0):
        """Compute overall trajectory reward"""
        # Reward successful goal achievement
        base_reward = 1.0 if success_score > 0.9 else 0.0

        # Penalize inefficient trajectories
        efficiency_penalty = 1.0 - (self.length / MAX_TRAJECTORY_LENGTH)
        efficiency_adjustment = efficiency_penalty * efficiency_score

        total_reward = base_reward * (1.0 + efficiency_adjustment)
        return total_reward

    def identify_long_horizon_challenges(self):
        """Flag trajectories with multi-step dependencies"""
        challenges = {
            "multi_step_dependencies": self.has_dependencies(),
            "trajectory_length": self.length,
            "tool_diversity": len(set(t["tool"] for t in self.turns)),
            "branching_required": self.has_conditional_logic()
        }
        return challenges
```

### Policy Gradient Update with Fine-Grained Rewards

Implement PG updates using turn-level advantages.

```python
# Policy gradient with fine-grained rewards
class FineGrainedPolicyGradient:
    def __init__(self, policy_model):
        self.policy = policy_model

    def update(self, trajectory, ground_truth, learning_rate=1e-3):
        """PG update with bipartite matching-based advantages"""
        # 1. Match predicted to ground truth
        matcher = BipartiteToolMatcher()
        matches = matcher.match_sequences(
            trajectory.extract_interactions(),
            ground_truth
        )

        # 2. Compute trajectory-level reward
        traj_reward = trajectory.compute_trajectory_reward(
            success_score=self.compute_success(trajectory, ground_truth)
        )

        # 3. Compute turn-level advantages
        turn_advantages = TurnLevelAdvantage().compute_turn_advantages(
            matches,
            traj_reward
        )

        # 4. Combine with trajectory-level signal
        dual_advantages = TurnLevelAdvantage().compute_dual_level_advantages(
            turn_advantages,
            traj_reward
        )

        # 5. Policy gradient update for each turn
        total_loss = 0.0
        for turn_idx, turn in enumerate(trajectory.turns):
            # Get log probability of selected action
            log_prob = self.policy.get_log_prob(turn["tool_name"], turn["tool_args"])

            # PG loss with advantage
            loss = -log_prob * dual_advantages[turn_idx]
            total_loss += loss

        # Optimization step
        avg_loss = total_loss / len(trajectory.turns)
        self.policy.backward(avg_loss, learning_rate)

        return avg_loss.item()

    def compute_success(self, trajectory, ground_truth):
        """Measure how well trajectory achieved goal"""
        if trajectory.extract_interactions() == ground_truth:
            return 1.0
        # Partial credit for mostly-correct trajectories
        correct_count = sum(
            1 for pred, truth in zip(trajectory.extract_interactions(), ground_truth)
            if pred == truth
        )
        return correct_count / len(ground_truth)
```

### Evaluation on Long-Horizon Tasks

Track improvements especially for complex multi-turn reasoning.

```python
# Long-horizon evaluation
class LongHorizonEval:
    def evaluate_model_size_advantage(self, model_4b, model_8b, test_set):
        """Compare small model with fine-grained rewards vs. larger baseline"""
        results = {
            "4b_accuracy": 0.0,
            "8b_accuracy": 0.0,
            "long_horizon_improvement": 0.0
        }

        long_horizon_tasks = [t for t in test_set if len(t["ground_truth"]) > 5]

        for task in long_horizon_tasks:
            # Evaluate 4B model
            pred_4b = model_4b.solve(task)
            correct_4b = self.is_correct(pred_4b, task)

            # Evaluate 8B baseline
            pred_8b = model_8b.solve(task)
            correct_8b = self.is_correct(pred_8b, task)

            if correct_4b and not correct_8b:
                results["long_horizon_improvement"] += 1

        results["4b_accuracy"] = sum(
            self.is_correct(model_4b.solve(t), t) for t in long_horizon_tasks
        ) / len(long_horizon_tasks)

        results["8b_accuracy"] = sum(
            self.is_correct(model_8b.solve(t), t) for t in long_horizon_tasks
        ) / len(long_horizon_tasks)

        return results

    def is_correct(self, prediction, task):
        """Check if prediction matches ground truth"""
        return prediction == task["ground_truth"]
```

## Performance Characteristics

- 4B models achieve competitive or superior performance vs. 8B baselines
- Particularly strong on long-horizon (multi-turn) tasks
- Fine-grained credit assignment improves learning efficiency
- Dual-level advantages balance local and global optimization

## Integration Pattern

1. Collect expert trajectories with annotated tool interactions
2. For each trajectory, match to ground truth using bipartite matching
3. Compute turn-level advantages from matches
4. Combine with trajectory-level rewards
5. Apply policy gradient updates using dual-level advantages
6. Evaluate especially on long-horizon, multi-tool tasks

## References

- Uniform trajectory rewards provide insufficient credit for multi-turn reasoning
- Bipartite matching enables optimal alignment of predicted to ground-truth interactions
- Turn-level granularity improves learning efficiency in complex tasks
- Small models with fine-grained supervision can outperform larger baselines
