---
name: regft-reference-guided-finetuning
title: "Learn Hard Problems During RL with Reference Guided Fine-tuning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.01223"
keywords: [Reinforcement Learning, Fine-Tuning, Mathematical Reasoning, Curriculum Learning, Reference Solutions]
description: "ReGFT pre-trains models on hybrid reference-augmented trajectories before RL, enabling them to solve harder problems and accelerate convergence."
---

# Technique: Reference-Guided Fine-Tuning for RL Pre-conditioning

Training language models for mathematical reasoning via pure RL often fails on hard problems—models lack sufficient prior knowledge to discover correct reasoning paths from random initialization. Conversely, pure imitation of human solutions fails when human proofs exceed the model's reasoning capacity. ReGFT bridges this gap by pre-training models on hybrid trajectories: partial human solutions combined with model-completed reasoning.

The key insight: models learn better when guided by human intuition on hard steps, then allowed to generate their own completions. This teaches reasoning patterns without requiring full solution imitation, preparing models to solve harder problems during subsequent RL training.

## Core Concept

The core insight is that ideal pre-training data lies between (1) pure imitation (too constraining) and (2) raw RL (insufficient guidance). ReGFT creates intermediate difficulty:

1. **Start with human solution**: Use expert-written partial solution
2. **Provide prefix**: Give model the first N steps
3. **Model completes**: Generate remaining steps in model's natural distribution
4. **Collect trajectory**: Create (partial_human_solution → model_completion) pairs
5. **Fine-tune on these**: Model learns to solve hard problems
6. **Start RL from checkpoint**: Model has learned basic patterns, ready to explore

This creates a curriculum where models gradually learn to handle harder problems.

## Architecture Overview

- **Reference Solution Pool**: Curated human-written solutions (e.g., AoPS)
- **Prefix Extraction**: Take first N steps as guidance
- **Model Completion**: Generate remaining steps conditioned on prefix
- **Trajectory Curation**: Keep only well-formed completions
- **SFT Phase**: Fine-tune model on (prefix → completion) pairs
- **RL Phase**: Continue from checkpoint with RL training

## Implementation Steps

ReGFT is a pre-training technique that prepares models for RL. Here's how to implement it:

Prepare a reference solution dataset and extract prefixes:

```python
import random
from typing import List, Dict, Tuple

class ReferenceGuidedPrep:
    """Prepares reference-guided fine-tuning data."""

    def __init__(self, reference_solutions: List[str], model_tokenizer):
        self.reference_solutions = reference_solutions
        self.tokenizer = model_tokenizer

    def extract_solution_steps(self, solution_text: str) -> List[str]:
        """
        Parse solution into logical steps.
        Example: splits by newlines or keywords like "Step 1", "Therefore", etc.
        """
        # Simple heuristic: split on blank lines or "Step N:" patterns
        import re

        steps = []
        current_step = ""

        for line in solution_text.split('\n'):
            line = line.strip()
            if re.match(r'Step \d+:', line) or (current_step and not line):
                if current_step:
                    steps.append(current_step)
                current_step = line if line else ""
            else:
                current_step += " " + line if current_step else line

        if current_step:
            steps.append(current_step)

        return steps

    def create_prefix_completion_pairs(
        self,
        max_prefix_ratio: float = 0.5,
    ) -> List[Dict]:
        """
        Create (prefix → completion) pairs from reference solutions.

        Args:
            max_prefix_ratio: Maximum fraction of solution to use as prefix (0.3-0.7)

        Returns:
            List of training examples
        """
        pairs = []

        for solution in self.reference_solutions:
            steps = self.extract_solution_steps(solution)

            if len(steps) < 2:
                continue  # Skip trivial solutions

            # Vary prefix length: 30%-70% of solution
            for prefix_ratio in [0.3, 0.5, 0.7]:
                prefix_len = max(1, int(len(steps) * prefix_ratio))
                prefix_steps = steps[:prefix_len]
                completion_steps = steps[prefix_len:]

                if not completion_steps:
                    continue  # Need at least one step to complete

                prefix_text = ' '.join(prefix_steps)
                completion_text = ' '.join(completion_steps)

                pairs.append({
                    'prefix': prefix_text,
                    'completion': completion_text,
                    'full_solution': solution,
                    'difficulty': 'hard' if len(steps) > 5 else 'medium',
                })

        return pairs
```

Implement model completion on prefixes:

```python
import torch

class ModelCompletionGenerator:
    """Generate model completions conditioned on reference prefixes."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_completion(
        self,
        prefix: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate continuation of a reference prefix.
        """
        # Tokenize prefix
        input_ids = self.tokenizer.encode(prefix, return_tensors='pt').to(self.device)

        # Generate completion
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode and extract new tokens only
        full_text = self.tokenizer.decode(output[0])
        completion = full_text[len(prefix):]

        return completion.strip()

    def create_training_pairs(
        self,
        prefix_completion_pairs: List[Dict],
        num_samples_per_prefix: int = 3,
    ) -> List[Dict]:
        """
        Generate multiple model completions for each prefix.
        """
        training_pairs = []

        for pair in prefix_completion_pairs:
            prefix = pair['prefix']

            for sample_idx in range(num_samples_per_prefix):
                # Generate completion
                completion = self.generate_completion(
                    prefix,
                    temperature=0.7 + 0.1 * sample_idx  # Vary temperature
                )

                training_pairs.append({
                    'prefix': prefix,
                    'model_completion': completion,
                    'reference_completion': pair['completion'],
                    'full_solution': pair['full_solution'],
                    'difficulty': pair['difficulty'],
                })

        return training_pairs
```

Implement SFT pre-training on curated pairs:

```python
class ReferenceGuidedPretraining:
    """
    Fine-tune model on reference-guided pairs before RL.
    """

    def __init__(self, model, tokenizer, learning_rate=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def create_sft_examples(
        self,
        training_pairs: List[Dict],
        filter_by_quality: bool = True,
    ) -> List[Dict]:
        """
        Create supervised fine-tuning examples.
        Optionally filter by completion quality.
        """
        sft_examples = []

        for pair in training_pairs:
            # Simple quality filter: completion not too different from reference
            if filter_by_quality:
                ref_len = len(pair['reference_completion'].split())
                completion_len = len(pair['model_completion'].split())

                # Accept if lengths similar (within 50%)
                if abs(completion_len - ref_len) > 0.5 * ref_len:
                    continue

            sft_examples.append({
                'input': pair['prefix'],
                'target': pair['model_completion'],  # Train on what model generated
            })

        return sft_examples

    def training_step(self, batch):
        """Single SFT training step."""
        inputs = batch['input']
        targets = batch['target']

        # Encode
        input_ids = self.tokenizer(
            inputs,
            padding=True,
            return_tensors='pt'
        ).input_ids.to(self.model.device)

        target_ids = self.tokenizer(
            targets,
            padding=True,
            return_tensors='pt'
        ).input_ids.to(self.model.device)

        # Forward pass
        outputs = self.model(input_ids, labels=target_ids)
        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        sft_examples: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 16,
    ):
        """Train on SFT examples."""
        for epoch in range(num_epochs):
            # Shuffle examples
            random.shuffle(sft_examples)

            # Create batches
            batches = [
                sft_examples[i:i+batch_size]
                for i in range(0, len(sft_examples), batch_size)
            ]

            total_loss = 0.0

            for batch in batches:
                # Format batch
                batch_dict = {
                    'input': [ex['input'] for ex in batch],
                    'target': [ex['target'] for ex in batch],
                }

                loss = self.training_step(batch_dict)
                total_loss += loss

            avg_loss = total_loss / len(batches)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        return self.model
```

Integration with RL training:

```python
def integrate_regft_with_rl(
    model,
    reference_solutions,
    tokenizer,
    num_sft_epochs=3,
):
    """
    Full pipeline: ReGFT pre-training → RL fine-tuning.
    """
    # Step 1: Prepare reference pairs
    prep = ReferenceGuidedPrep(reference_solutions, tokenizer)
    prefix_pairs = prep.create_prefix_completion_pairs(max_prefix_ratio=0.5)

    # Step 2: Generate model completions
    completion_gen = ModelCompletionGenerator(model, tokenizer)
    training_pairs = completion_gen.create_training_pairs(
        prefix_pairs,
        num_samples_per_prefix=2
    )

    # Step 3: Pre-train on curated pairs
    pretrain = ReferenceGuidedPretraining(model, tokenizer)
    sft_examples = pretrain.create_sft_examples(training_pairs, filter_by_quality=True)
    pretrained_model = pretrain.train(sft_examples, num_epochs=num_sft_epochs)

    # Step 4: Continue with RL training from pretrained checkpoint
    # (Pass pretrained_model to RL trainer)
    return pretrained_model
```

## Practical Guidance

**When to Use:**
- Mathematical reasoning tasks (AIME, BeyondAIME, competition math)
- When you have curated reference solutions available
- To enable models to tackle harder problems
- Before starting RL training on challenging benchmarks

**When NOT to Use:**
- Simple tasks where RL alone suffices
- When reference solutions are unavailable or low-quality
- Real-time systems (pre-training is offline)

**Data Preparation:**
- Collect 1K–10K high-quality reference solutions
- Clean and parse solutions into logical steps
- Vary prefix lengths (30%–70% typical)
- Filter low-quality model completions

**Hyperparameters:**
- `max_prefix_ratio`: 0.5 typical (half the solution as guidance)
- `temperature`: 0.6–0.8 for generation (lower = more deterministic)
- `num_epochs`: 2–5 for SFT pre-training
- `learning_rate`: 1e-5 to 5e-5

**Quality Filtering:**
- Keep completions within 50% of reference length
- Verify solutions are valid (if external solver available)
- Remove duplicates and near-duplicates

**Performance:**
- Increases solvable problem count (especially hard problems)
- Produces checkpoints that receive more positive RL rewards
- Accelerates DAPO/GRPO training convergence
- Improves final performance plateaus on reasoning benchmarks

---

**Reference:** [Learn Hard Problems During RL with Reference Guided Fine-tuning](https://arxiv.org/abs/2603.01223)
