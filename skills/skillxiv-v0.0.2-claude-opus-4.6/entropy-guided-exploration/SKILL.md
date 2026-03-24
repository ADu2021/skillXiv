---
name: entropy-guided-exploration
title: "First Return, Entropy-Eliciting Explore: FR3E Stable Reasoning in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07017"
keywords: [Reinforcement Learning, Reasoning Trajectories, Entropy-based Exploration, Policy Optimization, Mathematical Reasoning]
description: "Stabilize RL training on reasoning tasks by performing entropy-guided rollouts from uncertain decision points, avoiding policy collapse and premature convergence. Increases fully correct trajectories on math reasoning while maintaining stable entropy throughout training."
---

# Entropy-Guided Exploration: Structured Uncertainty-Driven Rollouts for Stable Reasoning

Reinforcement learning on mathematical reasoning with verifiable rewards (RLVR) is notoriously unstable—models converge prematurely to shallow tactics, ignore long chains of reasoning, or collapse into reward-gaming strategies. Traditional RL policies optimize based on final trajectory rewards without understanding where the model is uncertain about its reasoning path. FR3E solves this through two-phase structured exploration: first, generate base trajectories and identify high-entropy decision points (where the model is most uncertain); second, launch multiple exploratory rollouts from those uncertain states, gathering intermediate reward signals that guide the policy toward more robust reasoning.

When fine-tuning reasoning models on AIME, competition math, or other verifiable reasoning benchmarks, entropy-guided exploration prevents the catastrophic policy drift common in standard RL. By targeting exploration at moments of genuine uncertainty rather than applying uniform exploration pressure, the model learns to question weak assumptions rather than just generate longer outputs.

## Core Concept

FR3E operates in two synchronized phases. The First Return phase generates trajectories and computes token-level entropy across the sequence, identifying semantic decision points where entropy is high (model is uncertain). These positions become anchors for structured exploration. The Entropy-Eliciting Explore phase launches multiple diverse rollouts from each high-entropy state, simulating what would happen if the model had chosen differently at that critical juncture. Intermediate rewards from these partial trajectories inform the policy: if rollouts from a state tend to succeed, that state's decisions were good; if they fail, reconsider. Asymmetric clipping encourages the policy to explore beyond symmetric PPO bounds, and adaptive advantage modulation scales learning signals based on marginal value improvements between consecutive states, preventing both exploration collapse and reward-gaming.

## Architecture Overview

- **Base Trajectory Generator**: Produces initial reasoning chains conditioned on problem statements
- **Token-wise Entropy Computation**: Identifies decision points where model uncertainty is highest
- **Semantic Segmentation**: Groups entropy peaks into meaningful reasoning blocks
- **Rejection Sampling Filter**: Removes degenerate rollouts where all branches yield identical rewards
- **Multi-Rollout Explorer**: Generates diverse branches from high-entropy states via temperature sampling
- **Intermediate Reward Estimator**: Assigns partial credit based on continuation success
- **Asymmetric Policy Gradient**: Clip-Higher mechanism encouraging exploration without collapse

## Implementation

This example demonstrates entropy computation and identification of critical reasoning decision points in trajectories.

```python
# Entropy-guided exploration for reasoning trajectories
import torch
import torch.nn.functional as F
from collections import defaultdict

class EntropyGuidedExplorer:
    def __init__(self, model, tokenizer, temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def generate_base_trajectory(self, problem_statement, max_length=512):
        """Generate reasoning trajectory and compute token-level entropy."""

        # Tokenize problem
        input_ids = self.tokenizer(problem_statement, return_tensors='pt')['input_ids']

        # Generate trajectory with logits
        trajectories = []
        entropies = []
        token_ids = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # Last token logits

                # Compute entropy at this position
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                entropies.append(entropy.item())

                # Sample next token
                probs_normalized = probs / self.temperature if self.temperature > 0 else probs
                next_token = torch.multinomial(probs_normalized, num_samples=1)
                token_ids.append(next_token.item())

                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for end-of-sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        trajectory_text = self.tokenizer.decode(token_ids)
        return trajectory_text, torch.tensor(entropies)

    def identify_critical_decision_points(self, trajectory_text, entropies, threshold_percentile=75):
        """Identify high-entropy positions as semantic decision points."""

        # Compute threshold
        entropy_threshold = torch.quantile(entropies, threshold_percentile / 100.0)

        # Find positions with high entropy
        high_entropy_positions = torch.where(entropies > entropy_threshold)[0]

        # Group into semantic blocks (consecutive positions form one decision point)
        decision_blocks = []
        current_block = [high_entropy_positions[0].item()]

        for i in range(1, len(high_entropy_positions)):
            pos = high_entropy_positions[i].item()
            if pos - current_block[-1] <= 3:  # Within 3 tokens
                current_block.append(pos)
            else:
                decision_blocks.append(current_block)
                current_block = [pos]

        if current_block:
            decision_blocks.append(current_block)

        # Convert positions to text segments
        tokens = self.tokenizer.tokenize(trajectory_text)
        decision_points = []

        for block in decision_blocks:
            start_token = tokens[min(block)]
            avg_entropy = torch.mean(entropies[block])
            decision_points.append({
                'position': block[0],
                'segment': start_token,
                'entropy': avg_entropy.item(),
                'token_range': (block[0], block[-1])
            })

        return decision_points
```

This example shows the entropy-eliciting explore phase: launching diverse rollouts from critical decision points.

```python
def explore_from_decision_point(self, problem_statement, trajectory_prefix, decision_point_pos, num_rollouts=5):
    """Generate diverse rollouts from a critical decision point."""

    # Encode trajectory up to decision point
    prefix_ids = self.tokenizer(
        problem_statement + trajectory_prefix[:decision_point_pos],
        return_tensors='pt'
    )['input_ids']

    rollout_trajectories = []
    rollout_rewards = []

    for _ in range(num_rollouts):
        # Generate diverse continuation with higher temperature
        with torch.no_grad():
            outputs = self.model.generate(
                prefix_ids,
                max_length=512,
                temperature=self.temperature * 1.5,  # Higher temperature for diversity
                top_p=0.95,
                do_sample=True
            )

        rollout_text = self.tokenizer.decode(outputs[0])
        rollout_trajectories.append(rollout_text)

    return rollout_trajectories

def estimate_intermediate_reward(self, trajectory_continuation, verifier):
    """Estimate reward for partial trajectory using verifier."""

    try:
        # Extract intermediate answers from trajectory
        answers = self.extract_intermediate_answers(trajectory_continuation)

        if not answers:
            return 0.0

        # Check last extracted answer
        final_answer = answers[-1]
        reward = verifier.check_answer(final_answer)

        return reward

    except:
        return 0.0  # Failed parsing = no reward

def adaptive_advantage_modulation(self, states, state_values):
    """Scale learning signals based on marginal value improvements."""

    advantages = []

    for i in range(len(state_values) - 1):
        current_value = state_values[i]
        next_value = state_values[i + 1]

        # Marginal improvement: how much does value increase at next state?
        marginal_improvement = max(0, next_value - current_value)

        # Scale advantage by improvement (small improvements get lower weight)
        advantage = marginal_improvement * 10.0  # Scaling factor

        advantages.append(advantage)

    return torch.tensor(advantages)
```

This example demonstrates the complete FR3E training loop with asymmetric clipping and adaptive advantage scaling.

```python
class FR3ETrainer:
    def __init__(self, model, tokenizer, verifier, learning_rate=1e-5):
        self.explorer = EntropyGuidedExplorer(model, tokenizer)
        self.model = model
        self.verifier = verifier
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def training_step(self, problem_statement, ground_truth_answer):
        """Complete FR3E training step with entropy-guided exploration."""

        # Phase 1: Generate base trajectory
        trajectory, entropies = self.explorer.generate_base_trajectory(problem_statement)

        # Compute base reward
        base_reward = self.verifier.check_answer(
            self.extract_final_answer(trajectory)
        )

        # Phase 2: Identify critical decision points
        decision_points = self.explorer.identify_critical_decision_points(
            trajectory, entropies, threshold_percentile=75
        )

        # Phase 3: Explore from decision points
        all_rollout_rewards = []
        valid_decision_points = 0

        for decision_point in decision_points:
            pos = decision_point['position']

            # Generate rollouts from this decision point
            rollouts = self.explorer.explore_from_decision_point(
                problem_statement,
                trajectory,
                pos,
                num_rollouts=4
            )

            # Compute rewards for rollouts
            rollout_rewards = []
            for rollout in rollouts:
                reward = self.explorer.estimate_intermediate_reward(rollout, self.verifier)
                rollout_rewards.append(reward)

            # Rejection sampling: skip if all rollouts have identical rewards
            if len(set(rollout_rewards)) > 1:
                all_rollout_rewards.extend(rollout_rewards)
                valid_decision_points += 1

        # Phase 4: Compute advantages with adaptive modulation
        if valid_decision_points > 0:
            exploration_advantage = (
                sum(all_rollout_rewards) / len(all_rollout_rewards)
            ) - base_reward

            # Asymmetric clipping (Clip-Higher)
            ppo_clip_lower = 0.2
            ppo_clip_upper = 0.2
            clipped_advantage = torch.clamp(
                torch.tensor(exploration_advantage),
                -ppo_clip_lower,
                ppo_clip_upper
            ) if exploration_advantage < 0 else max(
                torch.tensor(exploration_advantage),
                torch.tensor(ppo_clip_upper)
            )
        else:
            clipped_advantage = torch.tensor(0.0)

        # Phase 5: Compute policy loss
        outputs = self.model(
            self.tokenizer(problem_statement, return_tensors='pt')['input_ids']
        )
        logits = outputs.logits

        # Log probability of trajectory
        log_prob = self._compute_trajectory_log_prob(
            self.tokenizer, trajectory, logits
        )

        # Policy gradient loss
        policy_loss = -log_prob * clipped_advantage

        # Entropy bonus to maintain exploration
        entropy_bonus = -0.01 * torch.mean(
            torch.distributions.Categorical(logits=logits).entropy()
        )

        total_loss = policy_loss + entropy_bonus

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'base_reward': base_reward,
            'exploration_advantage': exploration_advantage if valid_decision_points > 0 else 0.0,
            'valid_decision_points': valid_decision_points,
            'policy_loss': policy_loss.item(),
            'entropy_bonus': entropy_bonus.item(),
            'total_loss': total_loss.item()
        }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Entropy threshold percentile | 75 | Identify top 25% uncertain positions |
| Rollouts per decision point | 4-5 | Balance exploration diversity vs. compute |
| Rollout temperature | base_temp × 1.5 | Encourage diverse continuations |
| Entropy coefficient | 0.01 | Prevent policy collapse |
| PPO clip range (asymmetric) | -0.2 to +∞ | Allow exploration beyond upper bound |
| Advantage modulation scaling | 10.0 | Scale marginal improvements |
| Gradient clipping max norm | 1.0 | Prevent training instability |
| Trajectory max length | 512 tokens | Typical competition math reasoning |

**When to use:** Apply FR3E when training models on verifiable reasoning tasks—mathematics, logic puzzles, code correctness verification. Use when RL training is unstable and models collapse to shallow strategies. Ideal for problems where intermediate reasoning steps can be verified, not just final answers.

**When NOT to use:** Skip for tasks without verifiable intermediate steps (creative writing, open-ended reasoning). Avoid if computational budget is severely limited—multi-rollout exploration adds 4-5× overhead. Don't use for simple tasks where standard RL succeeds. Skip if your problem domain has sparse rewards and rarely verifiable intermediate results.

**Common pitfalls:** Setting entropy threshold too low (high percentile) includes all tokens, destroying decision point specificity. Too high discards useful exploration signals. Not using rejection sampling causes learning from degenerate rollouts where all branches have identical rewards. Forgetting asymmetric clipping negates exploration benefits—symmetric PPO prevents high advantages. Over-scaling with adaptive modulation can amplify noise. Not maintaining entropy bonus during RL causes policy collapse to deterministic tokens. Forgetting to verify that intermediate reward signals actually correlate with final correctness.

## Reference

FR3E Team. (2025). First Return, Entropy-Eliciting Explore: Stable Reasoning in LLMs. arXiv preprint arXiv:2507.07017. https://arxiv.org/abs/2507.07017
