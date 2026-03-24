---
name: fission-grpo-tool-use
title: "Robust Tool Use via Fission-GRPO"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.15625"
keywords: [Tool Use, Error Recovery, Reinforcement Learning, GRPO, Small Models]
description: "Train small models to recover from tool execution errors by treating errors as training signals. Use error simulators to generate recovery examples and fission failed trajectories into multiple recovery attempts."
---

# Robust Tool Use via Fission-GRPO

Small language models struggle when tools return errors: they often degenerate into repetitive invalid re-invocations without interpreting feedback or self-correcting. Standard RL treats errors as sparse negative signals, but vanishing gradients prevent learning recovery strategies. Fission-GRPO solves this by converting execution errors into multiplicative training signals: errors "fission" into parallel recovery attempts with diagnostic context, enabling the model to learn robust error recovery through on-policy RL.

The key insight is that errors are information: paired with correct recovery attempts, they provide supervised examples of corrective reasoning.

## Core Concept

Fission-GRPO uses a three-stage framework:

1. **Standard Exploration**: GRPO samples trajectories, computing rewards across format compliance, functional correctness, and efficiency
2. **Error Identification & Synthesis**: Flag failures and synthesize realistic error diagnostics via learned Error Simulator
3. **Fission-Based Updates**: Expand each error into G' parallel recovery attempts resampled on-policy, converting single failures into multiplicative training signals

This increases outcome diversity and restores meaningful within-group advantages even when most samples fail.

## Architecture Overview

- **GRPO Sampler**: Initial trajectory exploration with policy rollout
- **Error Detector**: Identifies non-compliant or incorrect outputs
- **Error Simulator**: Learned model generating realistic error messages from failed attempts
- **Recovery Resampler**: Resample multiple recovery trajectories conditioned on error
- **Advantage Computation**: Within-group advantage based on recovery success rates
- **Policy Update**: GRPO gradient steps using fissioned training signals

## Implementation

The method involves error detection, simulation, and fission-based resampling.

Implement error detection and categorization:

```python
import torch
from typing import List, Dict, Tuple

class ErrorDetector:
    """Identify and categorize execution errors."""

    def __init__(self, error_categories=None):
        self.categories = error_categories or [
            "format_error",
            "runtime_error",
            "logic_error",
            "timeout"
        ]

    def detect_error(self, output: str, api_response: str) -> Tuple[bool, str]:
        """Check if output is erroneous and categorize."""

        # Format check
        if not self.is_valid_format(output):
            return True, "format_error"

        # API response check
        if "Error" in api_response or "error" in api_response:
            return True, "runtime_error"

        # Logic validation
        if not self.validate_logic(output):
            return True, "logic_error"

        return False, "none"

    def is_valid_format(self, output: str) -> bool:
        """Validate output format (JSON, function call, etc.)."""
        try:
            json.loads(output)  # For JSON outputs
            return True
        except:
            return False

    def validate_logic(self, output: str) -> bool:
        """Check logical correctness of output."""
        # Task-specific validation
        return True

detector = ErrorDetector()
```

Implement error simulator for diagnostic generation:

```python
class ErrorSimulator:
    """Simulate realistic error messages for recovery training."""

    def __init__(self, model_name="Qwen3-32B", fine_tuned=True):
        self.model = load_model(model_name)
        if fine_tuned:
            self.model.load_lora("error_simulator_lora")

    def generate_diagnostic(self, input_query: str, failed_output: str,
                           error_category: str) -> str:
        """Generate realistic error message for failed attempt."""

        prompt = f"""You are an API error simulator. Given a failed attempt, generate a realistic error message.

Input: {input_query}
Failed output: {failed_output}
Error type: {error_category}

Generate a realistic error message that:
- Explains what went wrong
- Suggests how to fix it (without revealing correct answer)
- Mimics actual API error format

Error message:"""

        diagnostic = self.model.generate(prompt, max_tokens=100)
        return diagnostic.strip()

simulator = ErrorSimulator()
```

Implement fission-based trajectory expansion:

```python
def fission_grpo_training_step(policy_model, base_model, batch,
                               error_detector, error_simulator,
                               num_fission_attempts=4):
    """GRPO training with error fission for recovery."""

    results = []
    fissioned_trajectories = []

    # Stage 1: Standard exploration
    for query in batch:
        # Sample trajectory
        trajectory = policy_model.sample(query)
        output = trajectory["output"]
        api_response = trajectory["api_response"]

        # Evaluate
        is_error, error_type = error_detector.detect_error(output, api_response)

        # Compute reward
        reward = compute_reward(output, api_response, query)
        results.append({
            "trajectory": trajectory,
            "reward": reward,
            "is_error": is_error,
            "error_type": error_type
        })

        # Stage 2: Error fission if applicable
        if is_error:
            # Generate diagnostic
            diagnostic = error_simulator.generate_diagnostic(
                query, output, error_type
            )

            # Augment context with error
            augmented_query = f"{query}\n\nPrevious attempt failed with: {diagnostic}\n\nTry again:"

            # Stage 3: Fission - sample multiple recovery attempts
            for fission_idx in range(num_fission_attempts):
                recovery_trajectory = policy_model.sample(augmented_query)
                recovery_output = recovery_trajectory["output"]
                recovery_api_response = recovery_trajectory["api_response"]

                is_recovery_error, _ = error_detector.detect_error(
                    recovery_output, recovery_api_response
                )
                recovery_reward = compute_reward(
                    recovery_output, recovery_api_response, query
                )

                fissioned_trajectories.append({
                    "original_trajectory": trajectory,
                    "recovery_trajectory": recovery_trajectory,
                    "recovery_reward": recovery_reward,
                    "success": not is_recovery_error
                })

    # Compute advantages with fission
    advantages = compute_fissioned_advantages(results, fissioned_trajectories)

    # GRPO update
    loss = 0
    for i, result in enumerate(results):
        trajectory = result["trajectory"]
        advantage = advantages[i]

        # Policy gradient
        log_prob = compute_log_prob(policy_model, trajectory)
        loss += -(log_prob * advantage).mean()

        # Include fissioned trajectories from this sample
        matching_fissions = [f for f in fissioned_trajectories
                            if f["original_trajectory"] == trajectory]

        for fission in matching_fissions:
            recovery_log_prob = compute_log_prob(
                policy_model, fission["recovery_trajectory"]
            )
            fission_advantage = fission["recovery_reward"]
            loss += -(recovery_log_prob * fission_advantage).mean()

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()

def compute_fissioned_advantages(results, fissioned_trajectories):
    """Compute advantages accounting for fission expansion."""
    advantages = []

    for result in results:
        base_advantage = result["reward"]

        # Add fission bonus if error was caught and recovered
        matching_fissions = [f for f in fissioned_trajectories
                            if f["original_trajectory"] == result["trajectory"]]

        if matching_fissions:
            # Within-group advantage: how much recovery helps
            recovery_success_rate = sum(
                1 for f in matching_fissions if f["success"]
            ) / len(matching_fissions)

            # Boost advantage if recovery successful
            fission_bonus = recovery_success_rate * 0.5
            total_advantage = base_advantage + fission_bonus
        else:
            total_advantage = base_advantage

        advantages.append(total_advantage)

    return torch.tensor(advantages)
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Num Fission Attempts | 3-5 per error | Higher provides more training signal |
| Error Simulator Size | 32B fine-tuned | Balance realism and speed |
| Diagnostic Length | 50-150 tokens | Concise but informative |
| Fission Bonus Weight | 0.3-0.5 | Scale relative to base reward |
| Error Categories | 4-6 types | Cover main failure modes |
| Policy Model Size | 1.5B-8B (small) | Targets models struggling with errors |

**When to use**: For tool-using agents with APIs that return errors. For small models showing error degradation. When recovery learning is critical for success.

**When NOT to use**: For models with strong error handling (larger models often sufficient). When error messages are uninformative.

**Common pitfalls**:
- Error simulator must be realistic—validate diagnostics match actual API errors
- Too many fission attempts wastes compute—start with 3 and increase if needed
- Fission bonus weight can create strange behavior—monitor entropy and reward distributions
- Recovery context length matters—keep augmented queries reasonable
- Over-reliance on simulated errors can diverge from real errors—periodically validate on actual API

## Reference

Robust Tool Use via Fission-GRPO
https://arxiv.org/abs/2601.15625
