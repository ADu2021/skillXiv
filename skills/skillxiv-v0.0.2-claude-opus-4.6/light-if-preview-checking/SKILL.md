---
name: light-if-preview-checking
title: Light-IF - Preview and Self-Checking for Instruction Following
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03178
keywords: [instruction-following, self-checking, reasoning, data-curation]
description: "Multi-stage training approach using entropy-preserving SFT and token-wise entropy-adaptive RL to improve instruction adherence. Combines data curation with reward-guided reasoning, outperforming larger models on IFEval."
---

# Light-IF: Preview and Self-Checking for Instruction Following

## Core Concept

Light-IF addresses poor instruction adherence in language models by identifying the root cause: lazy reasoning during the thinking stage. The framework uses a multi-stage approach combining carefully curated data with entropy-adaptive reinforcement learning. By teaching models to preview instructions and self-check their work, the approach achieves strong instruction following without massive scale.

## Architecture Overview

- **Data Curation**: Complex instructions filtered into hard/easy/pass categories with rejection sampling
- **Entropy-Preserving SFT**: Supervised fine-tuning that maintains reasoning diversity
- **Token-wise Entropy-Adaptive RL (TEA-RL)**: Reinforcement learning guided by token-level entropy signals
- **Self-Checking Mechanism**: Models verify their own instruction adherence before finalizing outputs
- **Preview Stage**: Explicit instruction parsing and planning before response generation

## Implementation Steps

### Step 1: Curate Complex Instruction Dataset

Build high-quality dataset by filtering instructions by complexity and verifying solutions.

```python
from typing import List, Dict, Tuple
import random

class InstructionCurator:
    """
    Curate complex instruction dataset with difficulty categorization.
    """

    def __init__(self, classifier_model):
        self.classifier = classifier_model
        self.curated_dataset = []

    def filter_instructions(self, raw_instructions, solutions):
        """
        Filter instructions into difficulty categories.

        Args:
            raw_instructions: List of instruction strings
            solutions: List of (instruction_idx, solution) pairs

        Returns:
            Categorized dataset with hard/easy/pass labels
        """
        categorized = {"hard": [], "easy": [], "pass": []}

        for instruction, solution in zip(raw_instructions, solutions):
            # Analyze instruction complexity
            complexity_score = self._estimate_complexity(instruction)

            # Verify solution correctness via rejection sampling
            solution_quality = self._verify_solution(instruction, solution)

            if solution_quality < 0.5:
                # Solution doesn't properly follow instruction
                category = "pass"
            elif complexity_score > 0.7:
                # Complex instruction with good solution
                category = "hard"
            else:
                # Simple or medium instruction
                category = "easy"

            categorized[category].append({
                "instruction": instruction,
                "solution": solution,
                "complexity": complexity_score,
                "quality": solution_quality
            })

        return categorized

    def _estimate_complexity(self, instruction: str) -> float:
        """
        Estimate instruction complexity on 0-1 scale.

        Args:
            instruction: Instruction text

        Returns:
            Complexity score
        """
        # Multiple complexity heuristics
        complexity_signals = []

        # Signal 1: Instruction length
        length_score = min(len(instruction.split()), 100) / 100.0
        complexity_signals.append(length_score * 0.2)

        # Signal 2: Constraint count
        constraints = instruction.count("must") + instruction.count("cannot") + instruction.count("should")
        constraint_score = min(constraints, 5) / 5.0
        complexity_signals.append(constraint_score * 0.3)

        # Signal 3: Nested structure depth
        nesting = instruction.count("[") + instruction.count("(")
        nesting_score = min(nesting, 4) / 4.0
        complexity_signals.append(nesting_score * 0.3)

        # Signal 4: Semantic complexity (via model)
        model_complexity = self.classifier.predict_complexity(instruction)
        complexity_signals.append(model_complexity * 0.2)

        return sum(complexity_signals)

    def _verify_solution(self, instruction: str, solution: str) -> float:
        """
        Verify solution adherence to instruction.

        Args:
            instruction: Original instruction
            solution: Proposed solution

        Returns:
            Quality score 0-1 indicating instruction adherence
        """
        # Verify key requirements are met
        adherence_checks = []

        # Parse instruction constraints
        constraints = self._parse_constraints(instruction)

        for constraint in constraints:
            satisfied = self._check_constraint(solution, constraint)
            adherence_checks.append(satisfied)

        if not adherence_checks:
            return 1.0  # No constraints found, assume good

        quality = sum(adherence_checks) / len(adherence_checks)

        return quality

    def _parse_constraints(self, instruction: str) -> List[str]:
        """Extract constraints from instruction."""
        import re
        # Simple extraction; improve with NLP in practice
        must_pattern = r"(?:must|should|required to) ([^.!?]+)"
        constraints = re.findall(must_pattern, instruction)
        return constraints

    def _check_constraint(self, solution: str, constraint: str) -> bool:
        """Check if solution satisfies constraint."""
        # Implement constraint checking logic
        return constraint.lower() in solution.lower()

    def apply_rejection_sampling(self, dataset: Dict, target_quality: float = 0.8):
        """
        Apply rejection sampling to improve dataset quality.

        Args:
            dataset: Curated dataset
            target_quality: Target quality threshold

        Returns:
            Filtered dataset meeting quality threshold
        """
        filtered = {"hard": [], "easy": [], "pass": []}

        for category, examples in dataset.items():
            for example in examples:
                if example["quality"] >= target_quality:
                    filtered[category].append(example)

        return filtered
```

### Step 2: Implement Entropy-Preserving Supervised Fine-Tuning

Fine-tune with SFT while maintaining diversity in reasoning patterns.

```python
class EntropyPreservingSFT:
    """
    SFT that preserves reasoning diversity through entropy constraints.
    """

    def __init__(self, model, entropy_weight=0.1):
        self.model = model
        self.entropy_weight = entropy_weight

    def compute_entropy_loss(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization to prevent mode collapse.

        Args:
            model_outputs: Model logits [batch, seq_len, vocab_size]

        Returns:
            Entropy regularization loss
        """
        # Compute probability distribution
        probs = torch.softmax(model_outputs, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Target: maintain minimum entropy to preserve diversity
        min_target_entropy = 2.0  # Roughly 2 nats for diverse outputs

        entropy_loss = torch.clamp(min_target_entropy - entropy.mean(), min=0)

        return entropy_loss

    def sft_step(self, batch_instructions: List[str], batch_solutions: List[str]):
        """
        Perform SFT step with entropy preservation.

        Args:
            batch_instructions: Batch of instructions
            batch_solutions: Batch of correct solutions

        Returns:
            Loss metrics
        """
        # Prepare input-output pairs
        inputs = [f"Instruction: {inst}\nSolution:" for inst in batch_instructions]

        # Forward pass
        outputs = self.model(inputs)
        logits = outputs.logits

        # Compute standard SFT loss
        sft_loss = self.model.compute_language_modeling_loss(outputs, batch_solutions)

        # Compute entropy loss
        entropy_loss = self.compute_entropy_loss(logits)

        # Combined loss
        total_loss = sft_loss + self.entropy_weight * entropy_loss

        # Backward pass
        total_loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()

        return {
            "sft_loss": sft_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item()
        }

    def train_epoch(self, dataset: List[Dict], batch_size: int = 32):
        """
        Train for one epoch.

        Args:
            dataset: Training dataset with instruction-solution pairs
            batch_size: Batch size for training

        Returns:
            Average loss for epoch
        """
        total_loss = 0
        num_batches = 0

        for batch_start in range(0, len(dataset), batch_size):
            batch = dataset[batch_start:batch_start + batch_size]

            instructions = [ex["instruction"] for ex in batch]
            solutions = [ex["solution"] for ex in batch]

            metrics = self.sft_step(instructions, solutions)
            total_loss += metrics["total_loss"]
            num_batches += 1

        return total_loss / num_batches
```

### Step 3: Implement Token-wise Entropy-Adaptive RL

Create RL system that adapts learning by token-level entropy signals.

```python
class TokenwiseEntropyAdaptiveRL:
    """
    RL with token-wise entropy-adaptive reward scaling.
    """

    def __init__(self, model, instruction_reward_fn):
        self.model = model
        self.reward_fn = instruction_reward_fn

    def compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy at each token position.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]

        Returns:
            Token-level entropy [batch, seq_len]
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy

    def compute_adaptive_rewards(
        self,
        instruction: str,
        solution: str,
        logits: torch.Tensor,
        tokens: List[int]
    ) -> torch.Tensor:
        """
        Compute token-wise rewards adapted by entropy.

        Args:
            instruction: Input instruction
            solution: Generated solution
            logits: Model logits during generation
            tokens: Generated token IDs

        Returns:
            Token-wise rewards [seq_len]
        """
        # Compute instruction adherence reward
        adherence_reward = self.reward_fn.compute_adherence(instruction, solution)

        # Compute token entropy
        token_entropy = self.compute_token_entropy(logits)

        # Adaptive scaling: higher entropy tokens learn more
        entropy_scale = 1.0 + (token_entropy - token_entropy.mean()) / (token_entropy.std() + 1e-6)

        # Scale rewards by entropy
        adaptive_rewards = adherence_reward * entropy_scale.squeeze()

        return adaptive_rewards

    def rl_training_step(
        self,
        instructions: List[str],
        policy_samples: List[Dict],  # Contains generated solutions and logits
        learning_rate: float = 1e-5
    ):
        """
        Perform RL training step with token-wise entropy adaptation.

        Args:
            instructions: Batch of instructions
            policy_samples: Samples from model with associated logits
            learning_rate: Learning rate for optimizer

        Returns:
            Training metrics
        """
        total_policy_loss = 0
        num_samples = 0

        for instruction, sample in zip(instructions, policy_samples):
            solution = sample["solution"]
            logits = sample["logits"]
            log_probs = sample["log_probs"]

            # Compute adaptive rewards
            token_rewards = self.compute_adaptive_rewards(
                instruction,
                solution,
                logits,
                sample["tokens"]
            )

            # REINFORCE loss with adaptive rewards
            # Loss = -sum(log_probs * rewards)
            policy_loss = -(log_probs * token_rewards).sum()

            total_policy_loss += policy_loss
            num_samples += 1

        # Average and optimize
        avg_loss = total_policy_loss / max(num_samples, 1)

        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.model.optimizer.step()
        self.model.optimizer.zero_grad()

        return {
            "policy_loss": avg_loss.item(),
            "avg_adherence": token_rewards.mean().item()
        }
```

### Step 4: Implement Preview and Self-Checking

Add explicit preview and verification stages to instruction following.

```python
def generate_with_preview_and_checking(
    model,
    instruction: str,
    max_attempts: int = 3
) -> Tuple[str, bool]:
    """
    Generate response with preview and self-checking stages.

    Args:
        model: Language model
        instruction: User instruction
        max_attempts: Maximum refinement attempts

    Returns:
        (generated_response, meets_instruction_check)
    """
    # Stage 1: Preview - parse and plan
    preview_prompt = f"""
    Instruction: {instruction}

    Before responding, preview what this instruction requires:
    1. Main requirement:
    2. Key constraints:
    3. Output format:

    Preview:
    """

    preview = model.generate(preview_prompt, max_length=200)

    # Stage 2: Generate initial response
    generation_prompt = f"""
    Instruction: {instruction}

    Keep in mind the requirements:
    {preview}

    Response:
    """

    response = model.generate(generation_prompt, max_length=500)

    # Stage 3: Self-check - verify adherence
    check_prompt = f"""
    Instruction: {instruction}
    Response: {response}

    Does this response properly follow the instruction? Check:
    1. Does it meet the main requirement?
    2. Are all constraints satisfied?
    3. Is the output format correct?

    If any answer is NO, the response fails the check.
    Final verdict: PASS or FAIL
    """

    check_result = model.generate(check_prompt, max_length=100)

    passes_check = "PASS" in check_result.upper()

    # Stage 4: Refinement if needed
    attempts = 1
    while not passes_check and attempts < max_attempts:
        refinement_prompt = f"""
        Original instruction: {instruction}
        Previous response: {response}
        Check feedback: {check_result}

        Please refine the response to address the issues identified.
        Refined response:
        """

        response = model.generate(refinement_prompt, max_length=500)

        # Re-check
        check_result = model.generate(check_prompt.replace(response, response), max_length=100)

        passes_check = "PASS" in check_result.upper()

        attempts += 1

    return response, passes_check
```

## Practical Guidance

### When to Use Light-IF

- **Instruction following benchmarks**: IFEval, MTEval, complex prompt scenarios
- **Few-shot instruction learning**: Limited data for specific instruction patterns
- **Quality-focused applications**: Where correct instruction adherence is critical
- **Model scaling constraints**: Achieving strong performance without massive models

### When NOT to Use Light-IF

- **Simple instruction tasks**: Standard prompting may suffice
- **Real-time generation**: Multi-stage preview and checking adds latency
- **Minimal data availability**: Curation requires reasonable dataset size
- **Open-ended generation**: Self-checking works best with verifiable instruction requirements

### Hyperparameter Recommendations

- **Entropy weight in SFT**: 0.05-0.15 (balance diversity vs. instruction adherence)
- **Token entropy threshold**: 2.0 nats for diversity maintenance
- **RL learning rate**: 1e-5 to 5e-5 (conservative for stability)
- **Self-check attempts**: 2-3 (diminishing returns beyond)
- **Data quality threshold (rejection sampling)**: 0.75-0.85

### Key Insights

The critical insight is that instruction-following failures stem from lazy reasoning, not model incompetence. By making models explicitly preview instructions and self-check outputs, the approach forces genuine reasoning about requirements. Token-wise entropy adaptation prevents collapse to single reasoning patterns while focusing learning on uncertain tokens.

## Reference

**Light-IF: Preview and Self-Checking for Instruction Following** (arXiv:2508.03178)

Introduces entropy-preserving SFT and token-wise entropy-adaptive RL for instruction following. Outperforms larger models through systematic data curation, diverse reasoning, and self-checking mechanisms.
